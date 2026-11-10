import { useEffect, useState } from 'react';
import type { ModelInfo, Settings } from './api';
import { DEFAULT_SETTINGS, fetchModels } from './api';
import { ChatTab } from './components/ChatTab';
import { CompletionTab } from './components/CompletionTab';
import { InspectTab } from './components/InspectTab';
import { SettingsPanel } from './components/SettingsPanel';

type Tab = 'chat' | 'completion' | 'inspect';

const TABS: { key: Tab; label: string }[] = [
  { key: 'chat', label: 'Chat' },
  { key: 'completion', label: 'Completion' },
  { key: 'inspect', label: 'Inspect' },
];

function loadSettings(): Settings {
  try {
    return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem('llm-settings') ?? '{}') };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export default function App() {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>('chat');
  const [chatModel, setChatModel] = useState('chat');
  const [settings, setSettings] = useState<Settings>(loadSettings);

  useEffect(() => {
    fetchModels()
      .then((ms) => {
        setModels(ms);
        if (!ms.some((m) => m.key === 'chat') && ms[0]) setChatModel(ms[0].key);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('llm-settings', JSON.stringify(settings));
    } catch {
      /* storage unavailable: settings just won't persist */
    }
  }, [settings]);

  const current = models?.find((m) => m.key === chatModel);

  return (
    <div className="app">
      <header>
        <div>
          <h1>LLM Tester</h1>
          <p className="sub">
            Your 124M model, trained from scratch on Kaggle, running locally
            {models?.[0] && ` on ${models[0].device.toUpperCase()}`}.
          </p>
        </div>
        <nav>
          {TABS.map((t) => (
            <button key={t.key} className={tab === t.key ? 'active' : ''} onClick={() => setTab(t.key)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {error && (
        <div className="error">
          Can't reach the model server ({error}). Start it with <code>python webapp/server.py</code>.
        </div>
      )}
      {!models && !error && <div className="loading">Connecting to the model server…</div>}

      {models && (
        <div className="body">
          <main>
            {tab === 'chat' && (
              <>
                <div className="model-pick">
                  <select value={chatModel} onChange={(e) => setChatModel(e.target.value)}>
                    {models.map((m) => (
                      <option key={m.key} value={m.key}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                  {current && (
                    <span className="about">
                      {current.params_M}M params · {current.context}-token context · {current.about}
                    </span>
                  )}
                </div>
                <ChatTab key={chatModel} model={chatModel} settings={settings} />
              </>
            )}
            {tab === 'completion' && <CompletionTab models={models} settings={settings} />}
            {tab === 'inspect' && <InspectTab models={models} />}
          </main>
          {tab !== 'inspect' && <SettingsPanel value={settings} onChange={setSettings} />}
        </div>
      )}
    </div>
  );
}
