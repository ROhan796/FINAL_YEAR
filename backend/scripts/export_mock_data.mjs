#!/usr/bin/env node
/**
 * Exports the frontend's seed fixtures to JSON so the backend seeds the exact
 * same demo data the Next.js client renders today.
 *
 * Run from `backend/`:  node --experimental-strip-types scripts/export_mock_data.mjs
 * Output:               scripts/mock-data.json
 */

import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = path.resolve(here, '..', '..', 'lib', 'mock-data.ts');
const out = path.resolve(here, 'mock-data.json');

// Type stripping is only needed for the .ts fixture; Node handles it with the flag.
const mod = await import(`file:///${source.replace(/\\/g, '/')}`);

const payload = {};
for (const [key, value] of Object.entries(mod)) {
  if (Array.isArray(value)) payload[key] = value;
}

writeFileSync(out, JSON.stringify(payload, null, 2), 'utf8');
console.log(
  `wrote ${out}\n` +
    Object.entries(payload)
      .map(([k, v]) => `  ${k}: ${v.length}`)
      .join('\n'),
);
