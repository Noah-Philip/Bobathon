import './globals.css';
import type { Metadata } from 'next';
export const metadata: Metadata = { title: 'RampForge AI', description: 'From messy docs to new-hire ramp plans' };
export default function RootLayout({ children }: { children: React.ReactNode }) { return <html lang="en"><body>{children}</body></html>; }
