import { z } from 'zod';
import type { Chunk, Course, Lesson, Module, Objective, ObjectiveType, Project, Source } from './types';

const objectiveTypes = ['conceptual', 'procedural', 'troubleshooting', 'reference', 'project'] as const;
const ObjectiveSchema = z.object({ title: z.string().min(3), type: z.enum(objectiveTypes), difficulty: z.number().min(1).max(5) });
const SupportLevelSchema = z.preprocess(value => {
  if (typeof value !== 'string') return value;
  const normalized = value.toLowerCase();
  if (['strong', 'weak', 'insufficient'].includes(normalized)) return normalized;
  if (['beginner', 'intermediate', 'advanced'].includes(normalized)) return 'weak';
  return value;
}, z.enum(['strong', 'weak', 'insufficient']));
const QuizQuestionSchema = z.object({
  question: z.string(),
  options: z.array(z.string()),
  answer: z.string().optional(),
  correct: z.string().optional(),
  explanation: z.string().optional(),
}).transform(q => ({
  question: q.question,
  options: q.options,
  answer: q.answer ?? q.correct ?? q.options[0] ?? '',
  explanation: q.explanation ?? (q.answer || q.correct ? `Correct answer: ${q.answer ?? q.correct}` : 'Review the cited source material for the answer.'),
}));
const LessonSchema = z.object({
  title: z.string(),
  learningObjective: z.string(),
  explanation: z.string(),
  keyTerms: z.array(z.string()),
  example: z.string(),
  quiz: z.array(QuizQuestionSchema),
  task: z.object({ title: z.string(), instructions: z.string(), successCriteria: z.array(z.string()) }).optional(),
  optionalTask: z.string().optional(),
  citedChunkIds: z.array(z.string()),
  supportLevel: SupportLevelSchema,
}).transform(lesson => ({
  title: lesson.title,
  learningObjective: lesson.learningObjective,
  explanation: lesson.explanation,
  keyTerms: lesson.keyTerms,
  example: lesson.example,
  quiz: lesson.quiz,
  task: lesson.task ?? (lesson.optionalTask ? {
    title: 'Practice task',
    instructions: lesson.optionalTask,
    successCriteria: ['Complete the task using the cited source material.'],
  } : undefined),
  citedChunkIds: lesson.citedChunkIds,
  supportLevel: lesson.supportLevel,
}));
const OutlineSchema = z.object({ title: z.string(), modules: z.array(z.object({ title: z.string(), description: z.string(), objectiveIds: z.array(z.string()) })) });

const groqApiKey = () => process.env.GROQ_API_KEY;
export function hasLlmConfig() { return Boolean(groqApiKey()); }
export function getLlmSetupMessage() { return 'Generation requires a real Groq API key. Configure GROQ_API_KEY in .env.local, then restart the server.'; }
export const llmModel = () => process.env.GROQ_MODEL || 'llama-3.1-8b-instant';
const id = (p: string) => `${p}_${Math.random().toString(36).slice(2, 10)}`;

export function ingestSources(sources: Source[]) { return sources.filter(s => s.rawText.trim().length > 0); }

export function chunkSources(sources: Source[]): Chunk[] {
  const chunks: Chunk[] = [];
  for (const source of sources) {
    const lines = source.rawText.replace(/\r\n/g, '\n').split('\n');
    let section = 'Introduction'; let buffer: string[] = []; let n = 1;
    const flush = () => { const text = buffer.join('\n').trim(); if (text) chunks.push({ id: id('chunk'), sourceId: source.id, sourceTitle: source.title, sectionPath: section, locationLabel: `${source.title} · ${section} · chunk ${n++}`, text }); buffer = []; };
    for (const line of lines) {
      const heading = /^(#{1,6})\s+(.+)$/.exec(line.trim());
      if (heading) { flush(); section = heading[2]; continue; }
      buffer.push(line);
      if (buffer.join('\n').length > 1400 && line.trim() === '') flush();
    }
    flush();
  }
  return chunks.flatMap(c => c.text.length <= 1800 ? [c] : c.text.match(/[\s\S]{1,1600}(?:\s|$)/g)!.map((text, i) => ({ ...c, id: id('chunk'), text: text.trim(), locationLabel: `${c.locationLabel}.${i + 1}` })));
}

type GroqMessage = { role: 'system' | 'user'; content: string };

function groqEndpoint() {
  const key = groqApiKey();
  if (!key) throw new Error(getLlmSetupMessage());
  return { url: 'https://api.groq.com/openai/v1/chat/completions', key };
}

async function jsonChat<T>(messages: GroqMessage[], schema: z.ZodType<T, z.ZodTypeDef, unknown>) {
  const { url, key } = groqEndpoint();
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key}`,
    },
    body: JSON.stringify({
      model: llmModel(),
      messages,
      temperature: 0.2,
      response_format: { type: 'json_object' },
    }),
  });
  const body = await res.text();
  if (!res.ok) throw new Error(`Groq API request failed (${res.status}): ${body}`);
  let raw = '';
  try { raw = JSON.parse(body).choices?.[0]?.message?.content || ''; }
  catch (e) { throw new Error(`Groq API returned an unexpected response: ${(e as Error).message}\nRaw response: ${body}`); }
  try { return schema.parse(JSON.parse(raw)); } catch (e) { const err = new Error(`LLM returned invalid JSON: ${(e as Error).message}\nRaw response: ${raw}`); throw err; }
}

export async function extractObjectivesWithLLM(project: Project, chunks: Chunk[]): Promise<Objective[]> {
  const out: Objective[] = [];
  for (const chunk of chunks) {
    const data = await jsonChat([{ role: 'system', content: 'Extract 1-3 onboarding learning objectives from only the provided source chunk. Return JSON: {"objectives":[{"title":"...","type":"conceptual|procedural|troubleshooting|reference|project","difficulty":1}]}. Do not add facts not present.' }, { role: 'user', content: `Audience: ${project.audience}\nGoal: ${project.goal || 'N/A'}\nChunk ID: ${chunk.id}\nCitation: ${chunk.locationLabel}\nSource:\n${chunk.text}` }], z.object({ objectives: z.array(ObjectiveSchema).max(3) }));
    data.objectives.forEach(o => out.push({ id: id('obj'), projectId: project.id, title: o.title, type: o.type as ObjectiveType, difficulty: o.difficulty, evidenceChunkIds: [chunk.id] }));
  }
  return out;
}

const tokens = (s: string) => new Set(s.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(/\s+/).filter(Boolean));
const sim = (a: string, b: string) => { const A = tokens(a), B = tokens(b); const inter = [...A].filter(x => B.has(x)).length; return inter / Math.max(1, new Set([...A, ...B]).size); };
export function dedupeObjectives(objectives: Objective[]) { const kept: Objective[] = []; for (const o of objectives) { const m = kept.find(k => sim(k.title, o.title) > 0.72); if (m) m.evidenceChunkIds = [...new Set([...m.evidenceChunkIds, ...o.evidenceChunkIds])]; else kept.push(o); } return kept; }

export async function buildCourseOutlineWithLLM(project: Project, objectives: Objective[]): Promise<z.infer<typeof OutlineSchema>> {
  return jsonChat([{ role: 'system', content: 'Build an onboarding course outline from real extracted objectives. Order overview before details, setup before usage, concepts before procedures, common workflows before troubleshooting, easier before harder. Return JSON {"title":"...","modules":[{"title":"...","description":"...","objectiveIds":["..."]}]}.' }, { role: 'user', content: JSON.stringify({ project, suggestedModules: ['Overview','Setup','Core Concepts','Workflow','Testing and Debugging','First Practical Task'], objectives }) }], OutlineSchema);
}

export function retrieveEvidenceChunks(objective: Objective, chunks: Chunk[]) { return chunks.filter(c => objective.evidenceChunkIds.includes(c.id)); }

const lessonPrompt = 'Generate a short onboarding lesson using only the provided source chunks. Return only JSON with this exact shape: {"title":"...","learningObjective":"...","explanation":"...","keyTerms":["..."],"example":"...","quiz":[{"question":"...","options":["..."],"answer":"...","explanation":"..."}],"task":{"title":"...","instructions":"...","successCriteria":["..."]},"citedChunkIds":["chunk_id"],"supportLevel":"strong|weak|insufficient"}. Each quiz item must include answer and explanation. supportLevel must describe source support, not learner difficulty. If the sources do not contain enough information, say "Insufficient source material" rather than inventing content and set supportLevel to "insufficient".';

export async function generateLessonsWithLLM(objective: Objective, evidence: Chunk[]): Promise<Lesson> {
  const data = await jsonChat([{ role: 'system', content: lessonPrompt }, { role: 'user', content: JSON.stringify({ objective, sourceChunks: evidence.map(c => ({ id: c.id, citation: c.locationLabel, sourceTitle: c.sourceTitle, text: c.text })) }) }], LessonSchema);
  return { id: id('lesson'), objectiveId: objective.id, ...data, citations: data.citedChunkIds.map(cid => { const c = evidence.find(e => e.id === cid) || evidence[0]; return { chunkId: cid, sourceTitle: c?.sourceTitle || 'Unknown source', locationLabel: c?.locationLabel || cid, quote: c?.text.slice(0, 180) }; }) };
}

export async function runPipeline(project: Project, sources: Source[]): Promise<{ chunks: Chunk[]; objectives: Objective[]; course: Course; warnings: string[] }> {
  const chunks = chunkSources(ingestSources(sources));
  if (!chunks.length) throw new Error('Add at least one non-empty source before generating.');
  const objectives = dedupeObjectives(await extractObjectivesWithLLM(project, chunks));
  const outline = await buildCourseOutlineWithLLM(project, objectives);
  const modules: Module[] = [];
  for (const m of outline.modules) {
    const lessons = [];
    for (const oid of m.objectiveIds) { const obj = objectives.find(o => o.id === oid); if (obj) lessons.push(await generateLessonsWithLLM(obj, retrieveEvidenceChunks(obj, chunks))); }
    modules.push({ id: id('mod'), title: m.title, description: m.description, objectiveIds: m.objectiveIds, lessons });
  }
  return { chunks, objectives, course: { id: id('course'), projectId: project.id, title: outline.title, audience: project.audience, modules }, warnings: modules.flatMap(m => m.lessons).filter(l => l.supportLevel !== 'strong').map(l => `${l.title}: ${l.supportLevel} source support`) };
}
