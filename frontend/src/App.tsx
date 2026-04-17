import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { PaperPlaneRight, Microphone, NavigationArrow, Sparkle, Target, MapPin, ForkKnife, Bed, WarningCircle, CheckCircle } from '@phosphor-icons/react';
import { motion, AnimatePresence } from 'framer-motion';

type Stage = 'idle' | 'asr_loading' | 'extracting' | 'gathering' | 'done' | 'error';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content?: string;
  stage?: Stage;
  tags?: any;
  candidates?: any;
}

const api = axios.create({ baseURL: '/api' });

export default function App() {
  const [stage, setStage] = useState<Stage>('idle');
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [errorDetails, setErrorDetails] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, stage]);

  const addMessage = (msg: Message) => setMessages(prev => [...prev, msg]);
  const updateLastAssistantMessage = (updates: Partial<Message>) => {
    setMessages(prev => {
      const newList = [...prev];
      for (let i = newList.length - 1; i >= 0; i--) {
        if (newList[i].type === 'assistant') {
          newList[i] = { ...newList[i], ...updates };
          break;
        }
      }
      return newList;
    });
  };

  const handleRunPipeline = async (textToProcess: string) => {
    if (!textToProcess.trim()) return;

    try {
      // 1. Text entered. Start extraction directly.
      setStage('extracting');
      addMessage({ id: Date.now().toString(), type: 'assistant', stage: 'extracting' });

      const extractRes = await api.post('/extract/keywords', { text: textToProcess });
      const extractData = extractRes.data.data;
      
      updateLastAssistantMessage({ 
        content: `分析完成。去往[${extractData.destination || '未知'}]。预算 ${extractData.budget_min_cny || '?'} - ${extractData.budget_max_cny || '?'} 元。`,
        tags: extractData,
        stage: 'gathering' 
      });
      setStage('gathering');

      // 2. Gather candidates adapter
      const gatherRes = await api.post('/pipeline/gather-candidates', {
        extract_result_path: extractData.result_file_path,
        destination: extractData.destination
      });
      const finalData = gatherRes.data.data;

      updateLastAssistantMessage({
        stage: 'done',
        candidates: finalData.flattened_candidates
      });
      setStage('done');

    } catch (err: any) {
      console.error(err);
      setStage('error');
      setErrorDetails(err?.response?.data?.detail || err.message);
      updateLastAssistantMessage({ stage: 'error' });
    }
  };

  const handleSendText = () => {
    if (stage !== 'idle' && stage !== 'done' && stage !== 'error') return;
    if (!inputText.trim()) return;
    
    addMessage({ id: 'u' + Date.now().toString(), type: 'user', content: inputText });
    handleRunPipeline(inputText);
    setInputText('');
  };

  const handleAudioUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    addMessage({ id: 'u' + Date.now().toString(), type: 'user', content: `[音频文件: ${file.name}]` });
    addMessage({ id: 'a' + Date.now().toString(), type: 'assistant', stage: 'asr_loading' });
    setStage('asr_loading');

    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const asrRes = await api.post('/asr/transcribe', formData);
      const transcribedText = asrRes.data.data.text;
      
      updateLastAssistantMessage({ 
        content: `识别到语音内容：\n"${transcribedText}"`,
        stage: 'extracting' 
      });

      // Pass it to the next step
      handleRunPipeline(transcribedText);

    } catch (err: any) {
      console.error(err);
      setStage('error');
      setErrorDetails(err?.response?.data?.detail || err.message);
      updateLastAssistantMessage({ stage: 'error' });
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <Sparkle size={28} color="var(--accent-cyan)" weight="fill" />
        <h1>Intelligent Cockpit Travel Assistant</h1>
        <div className="assistant-status">
          {stage === 'idle' && 'READY'}
          {stage === 'asr_loading' && 'LISTENING...'}
          {stage === 'extracting' && 'ANALYZING...'}
          {stage === 'gathering' && 'GATHERING DATA...'}
          {stage === 'done' && 'COMPLETED'}
          {stage === 'error' && 'SYSTEM ERROR'}
        </div>
      </header>

      <main className="workspace">
        {/* Left / Top: Chat stream */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="stream-feed">
            {messages.length === 0 && (
              <div style={{ color: 'var(--text-secondary)', textAlign: 'center', margin: 'auto', opacity: 0.5 }}>
                <NavigationArrow size={48} weight="thin" style={{ marginBottom: '1rem' }} />
                <p>Describe your multi-modal travel request.</p>
                <p>Upload voice note or type below.</p>
              </div>
            )}
            <AnimatePresence>
              {messages.map((m) => (
                <motion.div 
                  key={m.id} 
                  className={`message ${m.type}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className={`message-icon ${m.type}`}>
                    {m.type === 'user' ? <Target size={20} /> : <Sparkle size={20} />}
                  </div>
                  <div className="message-content">
                    {m.content && <div className="message-body">{m.content}</div>}
                    
                    {/* Tags Extracted */}
                    {m.tags && Object.keys(m.tags).length > 0 && (
                      <div className="tags-container">
                        {m.tags.destination && (
                          <div className="tag"><MapPin size={14}/><span className="tag-label">Dest</span>{m.tags.destination}</div>
                        )}
                        {m.tags.people_count && (
                          <div className="tag"><span className="tag-label">Count</span>{m.tags.people_count}人</div>
                        )}
                        {m.tags.spot_keywords?.map((k: string) => (
                          <div key={k} className="tag"><span className="tag-label">Spot</span>{k}</div>
                        ))}
                        {m.tags.food_keywords?.map((k: string) => (
                          <div key={k} className="tag"><span className="tag-label">Food</span>{k}</div>
                        ))}
                      </div>
                    )}
                    
                    {/* Status Spinners inside Assistant message */}
                    {(m.stage === 'asr_loading' || m.stage === 'extracting' || m.stage === 'gathering') && (
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', marginTop: '0.8rem', color: 'var(--accent-cyan)' }}>
                        <div className="loading-dots">
                          <span></span><span></span><span></span>
                        </div>
                        <span style={{ fontSize: '0.85rem' }}>
                          {m.stage === 'asr_loading' && 'Transcribing audio signal...'}
                          {m.stage === 'extracting' && 'Extracting semantic nodes...'}
                          {m.stage === 'gathering' && 'Gathering holistic candidates via external sensors...'}
                        </span>
                      </div>
                    )}

                    {m.stage === 'error' && (
                      <div style={{ marginTop: '0.8rem', color: 'var(--accent-magenta)', fontSize: '0.9rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        <WarningCircle size={18} />
                        Pipeline failed: {errorDetails}
                      </div>
                    )}
                    
                    {m.stage === 'done' && (
                      <div style={{ marginTop: '0.8rem', color: '#00e676', fontSize: '0.9rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        <CheckCircle size={18} /> Candidates aggregated successfully.
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={endOfMessagesRef} />
          </div>

          <div className="chat-input-wrapper">
            <input 
              type="file" 
              accept="audio/*" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              onChange={handleAudioUpload}
            />
            <button 
              className="action-btn" 
              onClick={() => fileInputRef.current?.click()}
              disabled={stage !== 'idle' && stage !== 'done' && stage !== 'error'}
              title="Upload Audio"
            >
              <Microphone size={24} className={stage === 'asr_loading' ? 'mic-active' : ''} weight={stage === 'asr_loading' ? 'fill' : 'regular'} />
            </button>
            <input 
              className="chat-input" 
              placeholder="E.g., 这周末想去天津玩两三天..." 
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSendText()}
              disabled={stage !== 'idle' && stage !== 'done' && stage !== 'error'}
            />
            <button 
              className="action-btn primary" 
              onClick={handleSendText}
              disabled={(!inputText.trim() && stage === 'idle') || (stage !== 'idle' && stage !== 'done' && stage !== 'error')}
            >
              <PaperPlaneRight size={20} weight="bold" />
            </button>
          </div>
        </div>

        {/* Right / Bottom: Candidate Results Grid */}
        <div className="glass-panel" style={{ overflowY: 'auto' }}>
          <div className="section-title">
            <Sparkle size={20} weight="fill" color="var(--accent-cyan)"/> Optimized Trajectory Pool
          </div>
          
          <div className="candidates-grid">
            {messages.map(m => m.candidates && m.candidates.map((c: any) => (
              <div className="poi-card" key={c.poi_id + c.poi_type}>
                <div className="poi-header">
                  <div className="poi-name">{c.name}</div>
                  <div className="poi-type-badge">{c.poi_type}</div>
                </div>
                
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                  {c.poi_type === 'spot' && <MapPin size={16} color="var(--accent-cyan)"/>}
                  {c.poi_type === 'hotel' && <Bed size={16} color="var(--accent-purple)"/>}
                  {c.poi_type === 'food' && <ForkKnife size={16} color="#FF9900"/>}
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{c.address?.slice(0, 18)}...</span>
                </div>

                <div className="tags-container" style={{ marginTop: 0 }}>
                  {c.tags?.slice(0, 3).map((t: string) => (
                    <div className="tag" key={t} style={{ transform: 'scale(0.9)', margin: 0 }}>{t}</div>
                  ))}
                </div>

                <div className="poi-meta">
                  {c.rating && (
                    <div className="poi-stat highlight">
                      ★ {c.rating}
                    </div>
                  )}
                  {c.center_distance_m !== null && (
                    <div className="poi-stat">
                      <NavigationArrow size={14} /> {(c.center_distance_m / 1000).toFixed(1)} km
                    </div>
                  )}
                  {c.price_value_cny && (
                    <div className="poi-price" style={{ marginLeft: 'auto' }}>
                      <span>￥</span>{c.price_value_cny}
                    </div>
                  )}
                </div>
              </div>
            )))}
            
            {messages.every(m => !m.candidates) && stage !== 'gathering' && (
              <div style={{ color: 'var(--text-secondary)', padding: '2rem', textAlign: 'center', opacity: 0.5 }}>
                No active candidates.
              </div>
            )}
            {stage === 'gathering' && (
              <div style={{ display: 'flex', justifyContent: 'center', padding: '4rem 0', width: '100%', gridColumn: '1 / -1' }}>
                <div className="loading-dots"><span></span><span></span><span></span></div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
