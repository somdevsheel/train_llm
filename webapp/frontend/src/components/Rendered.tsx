import type { GenStats } from '../api';

/** Renders text with ```fenced``` blocks as code. Plain text only — no HTML injection. */
export function Rendered({ text }: { text: string }) {
  const parts = text.split('```');
  return (
    <>
      {parts.map((part, i) =>
        i % 2 === 1 ? (
          <pre key={i} className="code">
            {part.replace(/^[\w+-]*\n/, '').replace(/\n$/, '')}
          </pre>
        ) : part.trim() ? (
          // newlines next to a code fence would show as blank lines around the block
          <span key={i}>{parts.length > 1 ? part.replace(/^\n+|\n+$/g, '') : part}</span>
        ) : null,
      )}
    </>
  );
}

export function StatsLine({ stats }: { stats: GenStats }) {
  const why = { eos: 'finished', length: 'hit token limit', cancelled: 'stopped' }[stats.stop];
  return (
    <div className="stats">
      {stats.tokens} tokens · {stats.tok_per_s} tok/s · {stats.seconds}s
      {stats.prompt_tokens !== undefined &&
        ` · prompt ${stats.prompt_tokens}${stats.context ? `/${stats.context}` : ''} tokens`}{' '}
      · {why}
      {!!stats.dropped && (
        <span className="warn"> · {stats.dropped} older message{stats.dropped > 1 ? 's' : ''} dropped to fit the context</span>
      )}
    </div>
  );
}
