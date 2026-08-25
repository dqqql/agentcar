import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {
  Bed, CalendarBlank, CheckCircle, ClockCounterClockwise, Compass, ForkKnife,
  GearSix, Heart, ListChecks, MapPin, Microphone, NavigationArrow,
  PaperPlaneRight, Path, Question, Sparkle, Target, User, WarningCircle,
} from '@phosphor-icons/react';
import { AnimatePresence, motion } from 'framer-motion';
import emptyJourney from './assets/empty-journey.svg';
import sidebarJourney from './assets/sidebar-journey.svg';

type Stage = 'idle' | 'asr_loading' | 'extracting' | 'gathering' | 'done' | 'error';
type CandidateType = 'spot' | 'food' | 'hotel' | string;
type PageId = 'planning' | 'trips' | 'favorites' | 'history' | 'settings';

interface ExtractTags {
  destination?: string | null; dates?: string[]; budget_text?: string | null;
  budget_min_cny?: number | null; budget_max_cny?: number | null;
  people_count?: number | null; spot_keywords?: string[]; food_keywords?: string[];
  hotel_keywords?: string[]; travel_styles?: string[]; result_file_path?: string | null;
}

interface Candidate {
  poi_id: string; poi_type: CandidateType; name: string; address?: string | null;
  center_distance_m?: number | null; rating?: number | null; price_value_cny?: number | null;
  price_unit?: string | null; tags?: string[]; final_score?: number | null; rank?: number | null;
  review_count?: number | null; score_breakdown?: Record<string, number>;
}

interface ItineraryStop {
  poi_id: string; poi_type: CandidateType; name: string; address?: string | null;
  arrival_time?: string | null; departure_time?: string | null; visit_minutes?: number | null;
  travel_from_previous_minutes?: number | null; estimated_party_cost_cny?: number | null;
  candidate: Candidate;
}

interface ItineraryDay { day_index: number; stops: ItineraryStop[]; }

interface ItineraryRoute {
  route_id: string; route_type: 'matched' | 'relaxed' | 'budget'; title: string; summary: string;
  estimated_total_cost_cny?: number | null; estimated_travel_minutes?: number | null; total_score?: number | null;
  days: ItineraryDay[]; score_breakdown: Record<string, number>; warnings: string[];
  is_candidate_combination?: boolean;
}

interface Message {
  id: string; type: 'user' | 'assistant'; content?: string; stage?: Stage;
  tags?: ExtractTags; candidates?: Candidate[]; routes?: ItineraryRoute[];
}

const api = axios.create({ baseURL: '/api' });
const routeTypeLabels = { matched: '当前优先', relaxed: '次选组合', budget: '补充组合' } as const;
const poiTypeLabels: Record<string, string> = { spot: '景点', food: '餐饮', hotel: '酒店' };
const navigationItems = [
  { id: 'planning', label: '推荐规划', icon: Compass },
  { id: 'trips', label: '行程管理', icon: ListChecks },
  { id: 'favorites', label: '收藏路线', icon: Heart },
  { id: 'history', label: '历史记录', icon: ClockCounterClockwise },
  { id: 'settings', label: '设置中心', icon: GearSix },
] as const;

const tagNameMap: Record<string, string> = {
  air_conditioning: '空调', breakfast_option: '含早餐', front_desk_24h: '24 小时前台',
  business_area: '近商圈', restaurant: '餐厅', parking: '停车', wifi: 'Wi-Fi',
};

const formatBudget = (tags?: ExtractTags | null) => {
  if (!tags) return null;
  if (tags.budget_min_cny != null && tags.budget_max_cny != null) {
    return tags.budget_min_cny === tags.budget_max_cny
      ? `${tags.budget_min_cny} 元左右` : `${tags.budget_min_cny} - ${tags.budget_max_cny} 元`;
  }
  if (tags.budget_max_cny != null) return `${tags.budget_max_cny} 元以内`;
  return tags.budget_text || null;
};

const buildAnalysisSummary = (tags: ExtractTags) => {
  const parts: string[] = [];
  if (tags.destination) parts.push(`去往 ${tags.destination}`);
  const budget = formatBudget(tags);
  if (budget) parts.push(`预算 ${budget}`);
  if (tags.people_count) parts.push(`${tags.people_count} 人同行`);
  return parts.length ? `分析完成，${parts.join('，')}。` : '分析完成，已提取本次出行需求。';
};

const latestOf = <T,>(messages: Message[], pick: (message: Message) => T | undefined) => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const value = pick(messages[index]);
    if (value !== undefined) return value;
  }
  return undefined;
};

const getPoiIcon = (type: CandidateType) => {
  if (type === 'hotel') return <Bed size={17} />;
  if (type === 'food') return <ForkKnife size={17} />;
  return <MapPin size={17} />;
};

const visibleTags = (candidate: Candidate | null) => {
  if (!candidate) return [];
  return Array.from(new Set((candidate.tags || []).map((tag) => tagNameMap[tag.trim()]
    || (/[一-龥]/.test(tag) ? tag.trim() : null)).filter((tag): tag is string => Boolean(tag)))).slice(0, 3);
};

const stageText: Record<Stage, string> = {
  idle: '已连接', asr_loading: '语音处理中', extracting: '分析中',
  gathering: '路线生成中', done: '已完成', error: '系统错误',
};

const analysisSteps = [
  { title: '正在搜索中', description: '正在查找符合目的地与预算的出行内容' },
  { title: '筛选候选', description: '正在比较景点、餐饮和住宿候选' },
  { title: '汇总数据', description: '正在整理并生成完整路线方案' },
];

const wait = (milliseconds: number) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

const formatDistance = (distance?: number | null) => {
  if (distance == null) return null;
  return distance < 1000 ? `${Math.round(distance)} 米` : `${(distance / 1000).toFixed(1)} 公里`;
};

const formatCandidatePrice = (price?: number | null) => {
  if (price == null) return null;
  return `¥ ${Number(price.toFixed(1))}`;
};

const buildCandidateRoutes = (candidates: Candidate[]): ItineraryRoute[] => {
  if (!candidates.length) return [];
  const sorted = (type: CandidateType) => candidates
    .filter((candidate) => candidate.poi_type === type)
    .sort((left, right) => {
      const rankDelta = (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER);
      if (rankDelta !== 0) return rankDelta;
      return (right.final_score ?? -1) - (left.final_score ?? -1);
    });
  const groups = { spot: sorted('spot'), food: sorted('food'), hotel: sorted('hotel') };
  const patterns = [
    { spot: 0, food: 0, hotel: 0, type: 'matched' as const },
    { spot: 1, food: 1, hotel: 1, type: 'relaxed' as const },
    { spot: 2, food: 2, hotel: 2, type: 'budget' as const },
  ];

  return patterns.map((pattern, index) => {
    const selected = (['spot', 'food', 'hotel'] as const)
      .map((type) => groups[type][pattern[type]] || null)
      .filter((candidate): candidate is Candidate => Boolean(candidate));
    const stops: ItineraryStop[] = selected.map((candidate) => ({
      poi_id: candidate.poi_id,
      poi_type: candidate.poi_type,
      name: candidate.name,
      address: candidate.address,
      arrival_time: null,
      departure_time: null,
      visit_minutes: null,
      travel_from_previous_minutes: null,
      estimated_party_cost_cny: candidate.price_value_cny,
      candidate,
    }));
    const priced = selected.filter((candidate) => candidate.price_value_cny != null);
    const prototypePrice = priced.length
      ? priced.reduce((sum, candidate) => sum + (candidate.price_value_cny || 0), 0)
      : null;
    const scores = selected
      .map((candidate) => candidate.final_score)
      .filter((score): score is number => score != null);

    return {
      route_id: `candidate-combination-${index + 1}`,
      route_type: pattern.type,
      title: `候选组合 ${index + 1}`,
      summary: selected.length
        ? `由当前后端排序结果中的 ${selected.length} 个候选点组成，具体顺序与时间仍待规划。`
        : '当前类别候选不足，具体路线仍待规划。',
      estimated_total_cost_cny: prototypePrice,
      estimated_travel_minutes: null,
      total_score: scores.length ? scores.reduce((sum, score) => sum + score, 0) / scores.length : null,
      days: [{ day_index: 1, stops }],
      score_breakdown: {},
      warnings: ['当前仅为候选组合，并非正式行程。酒店价格、评分等为原型/模拟数据，请以实际预订信息为准。'],
      is_candidate_combination: true,
    };
  });
};

export default function App() {
  const [stage, setStage] = useState<Stage>('idle');
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [errorDetails, setErrorDetails] = useState('');
  const [selectedRouteId, setSelectedRouteId] = useState<string | null>(null);
  const [activePage, setActivePage] = useState<PageId>('planning');
  const [loadingPhraseIndex, setLoadingPhraseIndex] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);
  const detailScrollRef = useRef<HTMLDivElement>(null);
  const pipelineRunRef = useRef(0);

  const latestTags = latestOf(messages, (message) => message.tags) || null;
  const latestRoutes = latestOf(messages, (message) => message.routes) || [];
  const activeRoute = latestRoutes.find((route) => route.route_id === selectedRouteId) || latestRoutes[0] || null;
  const visibleMessages = messages.slice(-3);
  const canSubmit = ['idle', 'done', 'error'].includes(stage);
  const isProcessing = ['asr_loading', 'extracting', 'gathering'].includes(stage);
  const loadingSteps = stage === 'asr_loading'
    ? [{ title: '正在识别语音', description: '正在将语音转换为出行需求' }]
    : analysisSteps;
  const activeLoadingStep = loadingSteps[Math.min(loadingPhraseIndex, loadingSteps.length - 1)];

  useEffect(() => {
    if (isProcessing) endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
    if (stage === 'done') detailScrollRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  }, [messages, stage, isProcessing]);

  const addMessage = (message: Message) => setMessages((previous) => [...previous, message]);
  const updateLastAssistantMessage = (updates: Partial<Message>) => {
    setMessages((previous) => {
      const next = [...previous];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].type === 'assistant') { next[index] = { ...next[index], ...updates }; break; }
      }
      return next;
    });
  };

  const runPipeline = async (text: string) => {
    if (!text.trim()) return;
    const runId = pipelineRunRef.current + 1;
    pipelineRunRef.current = runId;
    setSelectedRouteId(null);
    setLoadingPhraseIndex(0);
    const progressSequence = (async () => {
      await wait(1000);
      if (pipelineRunRef.current !== runId) return;
      setLoadingPhraseIndex(1);
      await wait(1100);
      if (pipelineRunRef.current !== runId) return;
      setLoadingPhraseIndex(2);
      await wait(900);
    })();
    try {
      setStage('extracting');
      addMessage({ id: `a-${Date.now()}`, type: 'assistant', stage: 'extracting' });
      const extractResponse = await api.post('/extract/keywords', { text });
      const extractData: ExtractTags = extractResponse.data.data;
      const summary = buildAnalysisSummary(extractData);
      updateLastAssistantMessage({ content: summary, tags: extractData, stage: 'gathering' });
      setStage('gathering');
      const gatherResponse = await api.post('/pipeline/gather-candidates', {
        extract_result_path: extractData.result_file_path, destination: extractData.destination,
      });
      const rankedCandidates = gatherResponse.data.data.ranked_candidates
        || gatherResponse.data.data.flattened_candidates || [];
      const backendRoutes: ItineraryRoute[] = gatherResponse.data.data.routes || [];
      const routes = backendRoutes.length ? backendRoutes : buildCandidateRoutes(rankedCandidates);
      await progressSequence;
      if (pipelineRunRef.current !== runId) return;
      updateLastAssistantMessage({ content: summary, stage: 'done', candidates: rankedCandidates, routes });
      setStage('done');
    } catch (error: any) {
      pipelineRunRef.current += 1;
      setStage('error');
      setErrorDetails(error?.response?.data?.detail || error.message);
      updateLastAssistantMessage({ stage: 'error' });
    }
  };

  const sendText = () => {
    const text = inputText.trim();
    if (!text || !canSubmit) return;
    addMessage({ id: `u-${Date.now()}`, type: 'user', content: text });
    setInputText('');
    if (textAreaRef.current) textAreaRef.current.style.height = 'auto';
    void runPipeline(text);
  };

  const handleAudioUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    addMessage({ id: `u-${Date.now()}`, type: 'user', content: `[音频文件：${file.name}]` });
    addMessage({ id: `a-${Date.now()}`, type: 'assistant', stage: 'asr_loading' });
    setStage('asr_loading');
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await api.post('/asr/transcribe', formData);
      const text = response.data.data.text as string;
      updateLastAssistantMessage({ content: `已识别语音内容：\n${text}`, stage: 'extracting' });
      await runPipeline(text);
    } catch (error: any) {
      setStage('error');
      setErrorDetails(error?.response?.data?.detail || error.message);
      updateLastAssistantMessage({ stage: 'error' });
    } finally { event.target.value = ''; }
  };

  const composer = (
    <div className="composer">
      <input ref={fileInputRef} type="file" accept="audio/*" hidden onChange={handleAudioUpload} />
      <button className="icon-button" type="button" onClick={() => fileInputRef.current?.click()} disabled={!canSubmit} aria-label="上传音频">
        <Microphone size={20} weight={stage === 'asr_loading' ? 'fill' : 'regular'} />
      </button>
      <textarea ref={textAreaRef} value={inputText} rows={1} disabled={!canSubmit} aria-label="出行需求"
        placeholder="描述目的地、人数、预算和偏好…"
        onChange={(event) => {
          setInputText(event.target.value); event.target.style.height = 'auto';
          event.target.style.height = `${Math.min(event.target.scrollHeight, 96)}px`;
        }}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendText(); }
        }}
      />
      <button className="send-button" type="button" onClick={sendText} disabled={!inputText.trim() || !canSubmit} aria-label="发送需求">
        <PaperPlaneRight size={19} weight="bold" />
      </button>
    </div>
  );

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><MapPin size={21} weight="fill" /></span>
          <h1>智能旅途助手</h1>
          <span className={`connection connection--${stage}`}><i />{stageText[stage]}</span>
        </div>
        <button className="help-button" type="button"><Question size={17} />帮助与反馈</button>
      </header>

      <div className="app-body">
        <aside className="sidebar" aria-label="主导航">
          <nav>
            {navigationItems.map((item) => {
              const Icon = item.icon;
              return <button type="button" key={item.id}
                className={activePage === item.id ? 'nav-item nav-item--active' : 'nav-item'}
                onClick={() => setActivePage(item.id)} aria-label={item.label}
                aria-current={activePage === item.id ? 'page' : undefined}>
                <Icon size={20} /><span>{item.label}</span>
              </button>;
            })}
          </nav>
          <div className="sidebar-art" aria-hidden="true"><img src={sidebarJourney} alt="" /></div>
        </aside>

        {activePage !== 'planning' ? (
          <main className="placeholder-page">
            <img src={emptyJourney} alt="旅行地图与路线占位插画" />
            <p>{navigationItems.find((item) => item.id === activePage)?.label}功能正在准备中</p>
          </main>
        ) : (
          <main className="planning-page">
            <section className="route-rail" aria-labelledby="route-list-title">
              <div className="rail-heading">
                <h2 id="route-list-title">{latestRoutes.length ? `为你整理了 ${latestRoutes.length} 条路线` : '推荐路线'}</h2>
                {isProcessing && <span className="generating">分析中</span>}
              </div>
              <div className="route-list">
                {latestRoutes.map((route) => (
                  <button type="button" key={route.route_id}
                    className={activeRoute?.route_id === route.route_id ? 'route-card route-card--selected' : 'route-card'}
                    onClick={() => setSelectedRouteId(route.route_id)}>
                    <div className="route-card-heading">
                      <div><span>{route.title}</span><strong>{routeTypeLabels[route.route_type]}</strong></div>
                      {activeRoute?.route_id === route.route_id && <CheckCircle size={19} weight="fill" />}
                    </div>
                    <p>{route.summary}</p>
                    <div className="route-card-meta"><span><Path size={15} />{route.estimated_travel_minutes != null ? `${route.estimated_travel_minutes} 分钟交通` : '交通待规划'}</span>
                      <span>{route.estimated_total_cost_cny != null ? `参考价（原型）¥ ${Math.round(route.estimated_total_cost_cny)}` : '价格待确认'}</span>
                    </div>
                  </button>
                ))}
                {!latestRoutes.length && !isProcessing && (
                  <div className="rail-empty"><img src={emptyJourney} alt="" /><h3>路线生成后会显示在此处</h3></div>
                )}
                {isProcessing && (
                  <div className="rail-empty rail-empty--loading"><div className="loading-dots"><i /><i /><i /></div>
                    <h3>{activeLoadingStep.title}</h3>
                    <p>{activeLoadingStep.description}</p>
                  </div>
                )}
              </div>
            </section>

            <section className="detail-panel" aria-live="polite">
              <div className="detail-scroll" ref={detailScrollRef}>
                <AnimatePresence mode="wait" initial={false}>
                {activeRoute ? (
                  <motion.div className="result-content" key="result"
                    initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, ease: 'easeOut' }}>
                    <div className="detail-header"><div>
                      <h2>{activeRoute.title}</h2><p>{activeRoute.summary}</p></div>
                    </div>
                    <div className="trip-facts">
                      <div><CalendarBlank size={20} /><span><small>具体日期</small>{latestTags?.dates?.length ? latestTags.dates.join(' / ') : '待规划'}</span></div>
                      <div><User size={20} /><span><small>人数</small>{latestTags?.people_count ? `${latestTags.people_count} 人` : '待确认'}</span></div>
                      <div><span className="currency">¥</span><span><small>参考价格（原型）</small>{activeRoute.estimated_total_cost_cny != null ? `¥ ${Math.round(activeRoute.estimated_total_cost_cny)}` : '待确认'}</span></div>
                    </div>
                    <section className="overview-section">
                      <div className="section-label"><i />路线概览</div>
                      <p>{activeRoute.estimated_travel_minutes != null
                        ? `预计交通 ${activeRoute.estimated_travel_minutes} 分钟。时间为规划估算，出发前请核对营业时间。`
                        : '当前后端已完成候选排序，但尚未提供完整路线、时间段与交通耗时；以下内容按候选组合展示。'}</p>
                      <div className="itinerary-days">
                        {activeRoute.days.map((day) => (
                          <section className="itinerary-day" key={day.day_index}>
                            <h3>{activeRoute.is_candidate_combination ? '候选节点' : `第 ${day.day_index} 天`}</h3>
                            <div className="schedule-list">
                              {day.stops.map((stop) => {
                                const candidate = stop.candidate;
                                return <article className="schedule-stop" key={`${day.day_index}-${stop.poi_id}`}>
                                  <div className="schedule-time"><strong>{stop.arrival_time || '待规划'}</strong>{stop.departure_time && <span>{stop.departure_time}</span>}</div>
                                  <i aria-hidden="true" />
                                  <div className="schedule-card">
                                    <div className="stop-type">{getPoiIcon(stop.poi_type)}{poiTypeLabels[stop.poi_type] || stop.poi_type}</div>
                                    <h4>{stop.name}</h4>
                                    <p>{stop.address || '暂无地址信息'}</p>
                                    <div className="stop-meta">
                                      {stop.travel_from_previous_minutes != null && stop.travel_from_previous_minutes > 0 && <span>从上一站 {stop.travel_from_previous_minutes} 分钟</span>}
                                      {stop.visit_minutes != null ? <span>停留 {stop.visit_minutes} 分钟</span> : <span>停留时间待规划</span>}
                                      {candidate.rank != null && <span>同类排名 #{candidate.rank}</span>}
                                      {candidate.final_score != null && <span>匹配度 {Math.round(candidate.final_score * 100)}%</span>}
                                      {formatDistance(candidate.center_distance_m) && <span>距中心 {formatDistance(candidate.center_distance_m)}</span>}
                                      {candidate.rating != null && <span>评分（原型）★ {candidate.rating}</span>}
                                      {formatCandidatePrice(stop.estimated_party_cost_cny) && <span>价格（原型）{formatCandidatePrice(stop.estimated_party_cost_cny)}</span>}
                                    </div>
                                    {visibleTags(candidate).length > 0 && <div className="stop-tags">
                                      {visibleTags(candidate).map((tag) => <span key={tag}>{tag}</span>)}
                                    </div>}
                                  </div>
                                </article>;
                              })}
                            </div>
                          </section>
                        ))}
                      </div>
                      {activeRoute.warnings.length > 0 && <p className="route-warnings">提示：{activeRoute.warnings.join(' ')}</p>}
                    </section>
                  </motion.div>
                ) : isProcessing ? (
                  <motion.section className="process-panel" key="processing" layout
                    initial={{ opacity: 0, y: 18, minHeight: 260 }}
                    animate={{ opacity: 1, y: 0, minHeight: 138 }}
                    exit={{ opacity: 0, y: -8, minHeight: 0 }}
                    transition={{ duration: 0.32, ease: 'easeOut' }}>
                    <div className="loading-orbit" aria-hidden="true"><i /><span /><span /><span /></div>
                    <div className="process-copy">
                      <span>正在为你规划</span>
                      <AnimatePresence mode="wait">
                        <motion.h2 key={activeLoadingStep.title}
                          initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}>
                          {activeLoadingStep.title}
                        </motion.h2>
                      </AnimatePresence>
                      <p>{activeLoadingStep.description}</p>
                    </div>
                    <div className="process-steps" aria-label="分析进度">
                      {loadingSteps.map((step, index) => (
                        <span className={index === loadingPhraseIndex
                          ? 'process-step process-step--active'
                          : index < loadingPhraseIndex ? 'process-step process-step--complete' : 'process-step'} key={step.title}>
                          <i />{step.title}
                        </span>
                      ))}
                    </div>
                  </motion.section>
                ) : (
                  <motion.div className="welcome-panel" key="welcome"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, scale: 0.98 }}>
                    <span className="welcome-icon"><NavigationArrow size={31} weight="duotone" /></span>
                    <h2>说说你想去哪里</h2>
                    <p>可以输入文字，也可以上传一段语音。建议包含城市、人数、预算和偏好。</p>
                  </motion.div>
                )}
                </AnimatePresence>

                {visibleMessages.length > 0 && (
                  <section className="conversation-panel"><div className="section-label"><i />最近对话</div>
                    <AnimatePresence initial={false}>
                      {visibleMessages.map((message) => (
                        <motion.div className={`message message--${message.type}`} key={message.id}
                          initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}>
                          <span className="message-icon">{message.type === 'user' ? <Target size={16} /> : <Sparkle size={16} />}</span>
                          <div>{message.content && <p>{message.content}</p>}
                            {['asr_loading', 'extracting', 'gathering'].includes(message.stage || '') && <span className="message-state">{activeLoadingStep.title}…</span>}
                            {message.stage === 'done' && <span className="message-state message-state--success"><CheckCircle size={14} />路线已生成</span>}
                            {message.stage === 'error' && <span className="message-state message-state--error"><WarningCircle size={14} />{errorDetails}</span>}
                          </div>
                        </motion.div>
                      ))}
                    </AnimatePresence><div ref={endOfMessagesRef} />
                  </section>
                )}
              </div>
              <div className="composer-area">{composer}</div>
            </section>
          </main>
        )}
      </div>
    </div>
  );
}
