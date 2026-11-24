'use client';

import React, { Suspense, useState } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { Sparkles, ArrowRight, ShieldCheck, Lock, Mail } from 'lucide-react';
import { authApi } from '@/lib/api';
import { usePlatformStore } from '@/lib/platform-store';

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const nextPath = searchParams.get('next') || '/feed';
  const { retrySync } = usePlatformStore();

  const [email, setEmail] = useState('demo@ckgai.dev');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await authApi.login(email, password);
      await retrySync();
      router.push(nextPath);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Sign in failed');
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLogin = async () => {
    setError('');
    setLoading(true);
    try {
      await authApi.devLogin();
      await retrySync();
      router.push(nextPath);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Demo sign in failed — check backend connection');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f0eee6] flex items-center justify-center p-4 font-serif">
      <div className="w-full max-w-md bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-8 space-y-6 shadow-sm">
        <div className="space-y-2 text-center">
          <Link href="/" className="inline-flex items-center gap-2 no-underline">
            <div className="w-9 h-9 rounded-xl bg-[#141413] text-[#faf9f5] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-[#d97757]" />
            </div>
            <span className="font-sans text-lg font-bold text-[#141413] lowercase tracking-tight">
              suiiiiiiii · ckgai
            </span>
          </Link>
          <h1 className="text-2xl font-normal text-[#141413] pt-2">Sign in to your research workspace</h1>
          <p className="text-xs font-sans text-[#87867f]">
            Access your saved library, cross-domain knowledge graph, and AI research copilot.
          </p>
        </div>

        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-3 text-xs font-sans">
            {error}
          </div>
        )}

        <form onSubmit={handleLogin} className="space-y-4">
          <div className="space-y-1">
            <label className="font-sans text-xs font-semibold text-[#141413] flex items-center gap-1.5">
              <Mail className="w-3.5 h-3.5 text-[#87867f]" /> Institutional or Research Email
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-3 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413] focus:outline-none focus:border-[#d97757]"
            />
          </div>

          <div className="space-y-1">
            <label className="font-sans text-xs font-semibold text-[#141413] flex items-center gap-1.5">
              <Lock className="w-3.5 h-3.5 text-[#87867f]" /> Password
            </label>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter password (min 8 chars)"
              className="w-full p-3 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413] focus:outline-none focus:border-[#d97757]"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-clay py-3 text-xs font-sans flex items-center justify-center gap-2"
          >
            <span>{loading ? 'Signing in...' : 'Sign In'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        <div className="relative py-1">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-[#cccbc8]" />
          </div>
          <div className="relative flex justify-center text-[11px] font-sans uppercase">
            <span className="bg-[#faf9f5] px-2 text-[#87867f]">or quick evaluation</span>
          </div>
        </div>

        <button
          type="button"
          onClick={handleDemoLogin}
          disabled={loading}
          className="w-full btn-outlined-dark py-2.5 text-xs font-sans flex items-center justify-center gap-2"
        >
          <ShieldCheck className="w-4 h-4 text-[#d97757]" />
          <span>Continue as Verified Demo Researcher (Dr. Amara Nwosu)</span>
        </button>

        <p className="text-center font-sans text-xs text-[#87867f]">
          New researcher?{' '}
          <Link href="/register" className="text-[#d97757] font-semibold hover:underline">
            Create an account
          </Link>
        </p>
      </div>
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#f0eee6]" />}>
      <LoginForm />
    </Suspense>
  );
}
