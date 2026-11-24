'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Sparkles, ArrowRight, User, Building2, BookOpen, Mail, Lock } from 'lucide-react';
import { authApi } from '@/lib/api';
import { usePlatformStore } from '@/lib/platform-store';

export default function RegisterPage() {
  const router = useRouter();
  const { retrySync } = usePlatformStore();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [title, setTitle] = useState('Research Scientist');
  const [institution, setInstitution] = useState('');
  const [field, setField] = useState('Computer Science & AI');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await authApi.register({ name, email, password, title, institution, field });
      await retrySync();
      router.push('/feed');
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Registration failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#f0eee6] flex items-center justify-center p-4 font-serif">
      <div className="w-full max-w-lg bg-[#faf9f5] border border-[#cccbc8] rounded-[24px] p-8 space-y-6 shadow-sm">
        <div className="space-y-2 text-center">
          <Link href="/" className="inline-flex items-center gap-2 no-underline">
            <div className="w-9 h-9 rounded-xl bg-[#141413] text-[#faf9f5] flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-[#d97757]" />
            </div>
            <span className="font-sans text-lg font-bold text-[#141413] lowercase tracking-tight">
              suiiiiiiii · ckgai
            </span>
          </Link>
          <h1 className="text-2xl font-normal text-[#141413] pt-2">Create your researcher profile</h1>
          <p className="text-xs font-sans text-[#87867f]">
            Join the unified research graph — library, collaborators, AI copilot, and manuscript review.
          </p>
        </div>

        {error && (
          <div className="bg-rose-50 border border-rose-200 text-rose-800 rounded-xl p-3 text-xs font-sans">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-3.5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="font-sans text-xs font-semibold text-[#141413] flex items-center gap-1">
                <User className="w-3.5 h-3.5 text-[#87867f]" /> Full Name
              </label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Dr. Elena Vance"
                className="w-full p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413]"
              />
            </div>

            <div className="space-y-1">
              <label className="font-sans text-xs font-semibold text-[#141413] flex items-center gap-1">
                <Mail className="w-3.5 h-3.5 text-[#87867f]" /> Email
              </label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="elena@mit.edu"
                className="w-full p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413]"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="font-sans text-xs font-semibold text-[#141413] flex items-center gap-1">
                <Building2 className="w-3.5 h-3.5 text-[#87867f]" /> Institution
              </label>
              <input
                type="text"
                value={institution}
                onChange={(e) => setInstitution(e.target.value)}
                placeholder="MIT CSAIL"
                className="w-full p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413]"
              />
            </div>

            <div className="space-y-1">
              <label className="font-sans text-xs font-semibold text-[#141413] flex items-center gap-1">
                <BookOpen className="w-3.5 h-3.5 text-[#87867f]" /> Primary Field
              </label>
              <input
                type="text"
                value={field}
                onChange={(e) => setField(e.target.value)}
                className="w-full p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413]"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="space-y-1">
              <label className="font-sans text-xs font-semibold text-[#141413]">Academic Title</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413]"
              />
            </div>

            <div className="space-y-1">
              <label className="font-sans text-xs font-semibold text-[#141413] flex items-center gap-1">
                <Lock className="w-3.5 h-3.5 text-[#87867f]" /> Password (8+ chars)
              </label>
              <input
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full p-2.5 bg-[#f0eee6] border border-[#cccbc8] rounded-xl text-xs font-sans text-[#141413]"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full btn-clay py-3 text-xs font-sans flex items-center justify-center gap-2 mt-2"
          >
            <span>{loading ? 'Creating account...' : 'Create Researcher Account'}</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>

        <p className="text-center font-sans text-xs text-[#87867f]">
          Already registered?{' '}
          <Link href="/login" className="text-[#d97757] font-semibold hover:underline">
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
