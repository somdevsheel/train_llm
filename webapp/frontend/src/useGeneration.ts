import { useCallback, useRef, useState } from 'react';
import type { GenRequest, GenStats } from './api';
import { generate } from './api';

/** Runs one streaming generation at a time; `stop()` aborts it and keeps the partial text. */
export function useGeneration() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const ctrl = useRef<AbortController | null>(null);

  const run = useCallback(
    async (req: GenRequest, onToken: (t: string) => void): Promise<GenStats | null> => {
      ctrl.current?.abort();
      const c = new AbortController();
      ctrl.current = c;
      setBusy(true);
      setError(null);
      const t0 = performance.now();
      let tokens = 0;
      try {
        return await generate(req, (t) => { tokens += 1; onToken(t); }, c.signal);
      } catch (e) {
        if (c.signal.aborted) {
          const seconds = (performance.now() - t0) / 1000;
          return { tokens, seconds: +seconds.toFixed(2), tok_per_s: +(tokens / seconds).toFixed(1), stop: 'cancelled' };
        }
        setError(e instanceof Error ? e.message : String(e));
        return null;
      } finally {
        if (ctrl.current === c) {
          ctrl.current = null;
          setBusy(false);
        }
      }
    },
    [],
  );

  const stop = useCallback(() => ctrl.current?.abort(), []);
  return { run, stop, busy, error };
}
