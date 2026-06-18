import { NextResponse } from 'next/server';
import { runPipeline, getLlmSetupMessage, hasLlmConfig } from '@/lib/pipeline';
import type { Project, Source } from '@/lib/types';

export async function GET() { return NextResponse.json({ configured: hasLlmConfig(), message: hasLlmConfig() ? 'LLM configured' : getLlmSetupMessage(), model: process.env.OPENAI_MODEL || 'gpt-4.1-mini' }); }
export async function POST(req: Request) {
  try {
    if (!hasLlmConfig()) return NextResponse.json({ error: getLlmSetupMessage() }, { status: 400 });
    const body = await req.json() as { project: Project; sources: Source[] };
    const result = await runPipeline(body.project, body.sources);
    return NextResponse.json(result);
  } catch (e) { return NextResponse.json({ error: (e as Error).message }, { status: 500 }); }
}
