import { NextRequest, NextResponse } from 'next/server';

const PUBLIC_PATHS = new Set(['/', '/login', '/register']);

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  if (
    PUBLIC_PATHS.has(pathname) ||
    pathname.startsWith('/api/') ||
    pathname.startsWith('/_next/') ||
    pathname.includes('.')
  ) {
    return NextResponse.next();
  }

  const hasAuthCookie = req.cookies.get('ckgai_auth')?.value === '1';
  const explicitlyLoggedOut = req.cookies.get('ckgai_logged_out')?.value === '1';
  const strictAuth = process.env.NEXT_PUBLIC_REQUIRE_AUTH === 'true';

  if (explicitlyLoggedOut || (strictAuth && !hasAuthCookie)) {
    const loginUrl = req.nextUrl.clone();
    loginUrl.pathname = '/login';
    loginUrl.searchParams.set('next', pathname);
    return NextResponse.redirect(loginUrl);
  }

  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};
