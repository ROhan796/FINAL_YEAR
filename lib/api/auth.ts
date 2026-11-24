import { api, setTokens, clearTokens } from './client';

export interface UserProfile {
  id: string;
  name: string;
  title: string;
  institution: string;
  field: string;
  bio: string;
  avatar: string;
  verified: boolean;
  interests: string[];
  email: string;
}

export interface TokenPair {
  accessToken: string;
  refreshToken: string;
  user: UserProfile;
}

export interface AuthConfig {
  devAutoLogin: boolean;
  passwordAuth: boolean;
  environment: string;
}

export const authApi = {
  getConfig: () => api.get<AuthConfig>('/api/v1/auth/config'),

  devLogin: () =>
    api.post<TokenPair>('/api/v1/auth/dev-login').then((tokens) => {
      setTokens(tokens.accessToken, tokens.refreshToken);
      return tokens;
    }),

  login: (email: string, password: string) =>
    api.post<TokenPair>('/api/v1/auth/login', { email, password }).then((tokens) => {
      setTokens(tokens.accessToken, tokens.refreshToken);
      return tokens;
    }),

  register: (data: { email: string; password: string; name: string; title?: string; institution?: string; field?: string }) =>
    api.post<TokenPair>('/api/v1/auth/register', data).then((tokens) => {
      setTokens(tokens.accessToken, tokens.refreshToken);
      return tokens;
    }),

  logout: () => {
    clearTokens(true);
  },

  getMe: () => api.get<UserProfile>('/api/v1/users/me'),

  updateProfile: (data: Partial<UserProfile>) =>
    api.patch<UserProfile>('/api/v1/users/me', data),
};
