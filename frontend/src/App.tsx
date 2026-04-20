import { useEffect, useRef, useState } from 'react';
import axios from 'axios';
import {
  Bed,
  CheckCircle,
  ForkKnife,
  MapPin,
  Microphone,
  NavigationArrow,
  PaperPlaneRight,
  Sparkle,
  Target,
  WarningCircle,
  X,
} from '@phosphor-icons/react';
import { AnimatePresence, motion } from 'framer-motion';

type Stage = 'idle' | 'asr_loading' | 'extracting' | 'gathering' | 'done' | 'error';
type CandidateType = 'spot' | 'food' | 'hotel' | string;

interface ExtractTags {
  destination?: string | null;
  dates?: string[];
  budget_text?: string | null;
  budget_min_cny?: number | null;
  budget_max_cny?: number | null;
  people_count?: number | null;
  spot_keywords?: string[];
  food_keywords?: string[];
  hotel_keywords?: string[];
  travel_styles?: string[];
  result_file_path?: string | null;
}

interface Candidate {
  poi_id: string;
  poi_type: CandidateType;
  name: string;
  address?: string | null;
  center_distance_m?: number | null;
  rating?: number | null;
  price_value_cny?: number | null;
  tags?: string[];
  final_score?: number | null;
  rank?: number | null;
  score_breakdown?: Record<string, number>;
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content?: string;
  stage?: Stage;
  tags?: ExtractTags;
  candidates?: Candidate[];
}

interface RoutePlan {
  id: string;
  badge: string;
  title: string;
  subtitle: string;
  confidenceLabel: string;
  estimatedBudget: string | null;
  spot: Candidate | null;
  food: Candidate | null;
  hotel: Candidate | null;
}

const api = axios.create({ baseURL: '/api' });

const routeBadgeLabels = ['优先推荐', '平衡路线', '补充路线'];
const routeStopDefinitions = [
  { key: 'spot', label: '景点' },
  { key: 'food', label: '餐饮' },
  { key: 'hotel', label: '酒店' },
] as const;

const tagNameMap: Record<string, string> = {
  air_conditioning: '空调',
  breakfast_option: '含早餐',
  front_desk_24h: '24小时前台',
  business_area: '近商圈',
  restaurant: '餐厅',
  parking: '停车',
  wifi: 'Wi-Fi',
};

const getPoiIcon = (type: CandidateType) => {
  if (type === 'hotel') {
    return <Bed size={16} color="var(--accent-purple)" />;
  }
  if (type === 'food') {
    return <ForkKnife size={16} color="#FFB347" />;
  }
  return <MapPin size={16} color="var(--accent-cyan)" />;
};

const formatCandidateTag = (tag: string) => {
  const trimmed = tag.trim();
  if (!trimmed) {
    return null;
  }
  if (tagNameMap[trimmed]) {
    return tagNameMap[trimmed];
  }
  if (/[\u4e00-\u9fa5]/.test(trimmed)) {
    return trimmed;
  }
  return null;
};

const getVisibleCandidateTags = (candidate: Candidate | null, limit = 2) => {
  if (!candidate) {
    return [];
  }

  const tags = (candidate.tags || [])
    .map(formatCandidateTag)
    .filter((tag): tag is string => Boolean(tag));

  const uniqueTags = Array.from(new Set(tags));
  if (uniqueTags.length > 0) {
    return uniqueTags.slice(0, limit);
  }

  if (candidate.poi_type === 'hotel') {
    return ['住宿候选'];
  }
  if (candidate.poi_type === 'food') {
    return ['餐饮候选'];
  }
  return ['景点候选'];
};

const formatDistance = (distance?: number | null) => {
  if (distance == null) {
    return null;
  }
  if (distance < 1000) {
    return `${distance} 米`;
  }
  return `${(distance / 1000).toFixed(1)} 公里`;
};

const formatPrice = (price?: number | null) => {
  if (price == null) {
    return null;
  }
  const rounded = Number(price.toFixed(1));
  return Number.isInteger(rounded) ? `${rounded}` : `${rounded}`;
};

const formatBudget = (tags?: ExtractTags) => {
  if (!tags) {
    return null;
  }
  if (tags.budget_min_cny != null && tags.budget_max_cny != null) {
    if (tags.budget_min_cny === tags.budget_max_cny) {
      return `${tags.budget_min_cny} 元左右`;
    }
    return `${tags.budget_min_cny} - ${tags.budget_max_cny} 元`;
  }
  if (tags.budget_max_cny != null) {
    return `${tags.budget_max_cny} 元以内`;
  }
  if (tags.budget_text) {
    return tags.budget_text;
  }
  return null;
};

const buildAnalysisSummary = (tags: ExtractTags) => {
  const parts: string[] = [];

  if (tags.destination) {
    parts.push(`去往 ${tags.destination}`);
  }

  const budget = formatBudget(tags);
  if (budget) {
    parts.push(`预算 ${budget}`);
  }

  if (tags.people_count) {
    parts.push(`${tags.people_count} 人同行`);
  }

  return parts.length > 0 ? `分析完成，${parts.join('，')}。` : '分析完成，已提取本次出行需求。';
};

const getLatestUserMessage = (messages: Message[]) =>
  [...messages].reverse().find((message) => message.type === 'user')?.content || '';

const getLatestTags = (messages: Message[]) =>
  [...messages].reverse().find((message) => message.tags)?.tags || null;

const getLatestCandidates = (messages: Message[]) =>
  [...messages].reverse().find((message) => message.candidates)?.candidates || [];

const getVisibleMessages = (messages: Message[]) => messages.slice(-4);

const getCandidateScore = (candidate: Candidate, tags: ExtractTags | null) => {
  const ratingScore = (candidate.rating ?? 3.8) * 14;
  const distanceScore =
    candidate.center_distance_m == null
      ? 4
      : Math.max(0, 8000 - Math.min(candidate.center_distance_m, 8000)) / 800;

  const budgetMax = tags?.budget_max_cny ?? null;
  const price = candidate.price_value_cny ?? null;
  const priceScore =
    price == null
      ? 4
      : budgetMax == null
        ? Math.max(0, 1200 - Math.min(price, 1200)) / 220
        : price <= budgetMax
          ? 5
          : Math.max(0, 5 - (price - budgetMax) / Math.max(budgetMax, 1));

  return ratingScore + distanceScore + priceScore;
};

const getDisplayCandidateScore = (candidate: Candidate, tags: ExtractTags | null) =>
  candidate.final_score != null ? candidate.final_score * 100 : getCandidateScore(candidate, tags);

const getRouteStops = (route: RoutePlan) =>
  routeStopDefinitions.map((item) => ({
    ...item,
    candidate: route[item.key],
  }));

const pickCandidate = (items: Candidate[], preferredIndex: number) => {
  if (items.length === 0) {
    return null;
  }
  return items[Math.min(preferredIndex, items.length - 1)];
};

const buildRoutePlans = (candidates: Candidate[], tags: ExtractTags | null): RoutePlan[] => {
  if (candidates.length === 0) {
    return [];
  }

  const grouped = {
    spot: candidates.filter((candidate) => candidate.poi_type === 'spot'),
    food: candidates.filter((candidate) => candidate.poi_type === 'food'),
    hotel: candidates.filter((candidate) => candidate.poi_type === 'hotel'),
  };

  const hasBackendRanking = candidates.some((candidate) => candidate.final_score != null);

  const sortByScore = (items: Candidate[]) => {
    if (hasBackendRanking) {
      return [...items].sort((left, right) => {
        const scoreDelta = (right.final_score ?? -1) - (left.final_score ?? -1);
        if (scoreDelta !== 0) {
          return scoreDelta;
        }
        return (left.rank ?? Number.MAX_SAFE_INTEGER) - (right.rank ?? Number.MAX_SAFE_INTEGER);
      });
    }
    return [...items].sort((left, right) => getCandidateScore(right, tags) - getCandidateScore(left, tags));
  };

  const ranked = {
    spot: sortByScore(grouped.spot),
    food: sortByScore(grouped.food),
    hotel: sortByScore(grouped.hotel),
  };

  const selectionPatterns = [
    { spot: 0, food: 0, hotel: 0 },
    { spot: 1, food: 1, hotel: 1 },
    { spot: 2, food: 0, hotel: 2 },
  ];

  return selectionPatterns.map((pattern, index) => {
    const spot = pickCandidate(ranked.spot, pattern.spot);
    const food = pickCandidate(ranked.food, pattern.food);
    const hotel = pickCandidate(ranked.hotel, pattern.hotel);
    const availableItems = [spot, food, hotel].filter((item): item is Candidate => Boolean(item));
    const filledCount = availableItems.length;
    const averageScore =
      availableItems.length > 0
        ? availableItems.reduce((sum, item) => sum + getDisplayCandidateScore(item, tags), 0) / availableItems.length
        : 0;
    const estimatedBudgetValue = availableItems.reduce((sum, item) => sum + (item.price_value_cny ?? 0), 0);

    return {
      id: `route-${index + 1}`,
      badge: routeBadgeLabels[index] || `路线 ${index + 1}`,
      title: `路线 ${index + 1}`,
      subtitle:
        filledCount === 3
          ? '景点、餐饮、酒店都已补齐'
          : filledCount === 2
            ? '已组合出 2 个节点，剩余节点先占位'
            : '先展示当前可用节点',
      confidenceLabel:
        averageScore >= 55 ? '匹配度高' : averageScore >= 48 ? '匹配度中' :'路线简介',
      estimatedBudget: estimatedBudgetValue > 0 ? `参考消费 ¥${Math.round(estimatedBudgetValue)}` : null,
      spot,
      food,
      hotel,
    };
  });
};

export default function App() {
  const [stage, setStage] = useState<Stage>('idle');
  const [inputText, setInputText] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [errorDetails, setErrorDetails] = useState('');
  const [selectedRoute, setSelectedRoute] = useState<RoutePlan | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  const latestTags = getLatestTags(messages);
  const latestCandidates = getLatestCandidates(messages);
  const latestRoutes = buildRoutePlans(latestCandidates, latestTags);
  const latestUserMessage = getLatestUserMessage(messages);
  const visibleMessages = getVisibleMessages(messages);
  const isInputFocus = messages.length === 0 && stage === 'idle';
  const workspaceClassName = 'workspace workspace--stable';
  const resultHint =
    stage === 'gathering'
      ? '正在生成展示路线...'
      : latestRoutes.length > 0
        ? `已整理 ${latestRoutes.length} 条展示路线`
        : '展示路线将在这里生成';

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, stage]);

  useEffect(() => {
    if (!selectedRoute) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setSelectedRoute(null);
      }
    };

    document.body.style.overflow = 'hidden';
    window.addEventListener('keydown', handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedRoute]);

  const resetComposerHeight = () => {
    if (!textAreaRef.current) {
      return;
    }
    textAreaRef.current.style.height = 'auto';
  };

  const syncComposerHeight = () => {
    if (!textAreaRef.current) {
      return;
    }
    textAreaRef.current.style.height = 'auto';
    textAreaRef.current.style.height = `${Math.min(textAreaRef.current.scrollHeight, 120)}px`;
  };

  const addMessage = (message: Message) => {
    setMessages((previous) => [...previous, message]);
  };

  const updateLastAssistantMessage = (updates: Partial<Message>) => {
    setMessages((previous) => {
      const nextMessages = [...previous];
      for (let index = nextMessages.length - 1; index >= 0; index -= 1) {
        if (nextMessages[index].type === 'assistant') {
          nextMessages[index] = { ...nextMessages[index], ...updates };
          break;
        }
      }
      return nextMessages;
    });
  };

  const handleRunPipeline = async (textToProcess: string) => {
    if (!textToProcess.trim()) {
      return;
    }

    try {
      setStage('extracting');
      addMessage({ id: `${Date.now()}`, type: 'assistant', stage: 'extracting' });

      const extractResponse = await api.post('/extract/keywords', { text: textToProcess });
      const extractData: ExtractTags = extractResponse.data.data;
      const assistantSummary = buildAnalysisSummary(extractData);

      updateLastAssistantMessage({
        content: assistantSummary,
        tags: extractData,
        stage: 'gathering',
      });
      setStage('gathering');

      const gatherResponse = await api.post('/pipeline/gather-candidates', {
        extract_result_path: extractData.result_file_path,
        destination: extractData.destination,
      });
      const rankedCandidates =
        gatherResponse.data.data.ranked_candidates || gatherResponse.data.data.flattened_candidates || [];

      updateLastAssistantMessage({
        content: assistantSummary,
        stage: 'done',
        candidates: rankedCandidates,
      });
      setStage('done');
    } catch (error: any) {
      console.error(error);
      setStage('error');
      setErrorDetails(error?.response?.data?.detail || error.message);
      updateLastAssistantMessage({ stage: 'error' });
    }
  };

  const handleSendText = () => {
    if (!inputText.trim()) {
      return;
    }
    if (!['idle', 'done', 'error'].includes(stage)) {
      return;
    }

    addMessage({ id: `u${Date.now()}`, type: 'user', content: inputText.trim() });
    handleRunPipeline(inputText.trim());
    setInputText('');
    resetComposerHeight();
  };

  const handleAudioUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    addMessage({ id: `u${Date.now()}`, type: 'user', content: `[音频文件：${file.name}]` });
    addMessage({ id: `a${Date.now()}`, type: 'assistant', stage: 'asr_loading' });
    setStage('asr_loading');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const asrResponse = await api.post('/asr/transcribe', formData);
      const transcribedText = asrResponse.data.data.text as string;

      updateLastAssistantMessage({
        content: `已识别语音内容：\n${transcribedText}`,
        stage: 'extracting',
      });

      handleRunPipeline(transcribedText);
    } catch (error: any) {
      console.error(error);
      setStage('error');
      setErrorDetails(error?.response?.data?.detail || error.message);
      updateLastAssistantMessage({ stage: 'error' });
    } finally {
      event.target.value = '';
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <Sparkle size={28} color="var(--accent-cyan)" weight="fill" />
        <h1>智能座舱出行助手</h1>
        <div className="assistant-status">
          {stage === 'idle' && '就绪'}
          {stage === 'asr_loading' && '语音处理中'}
          {stage === 'extracting' && '分析中'}
          {stage === 'gathering' && '路线生成中'}
          {stage === 'done' && '已完成'}
          {stage === 'error' && '系统错误'}
        </div>
      </header>

      <main className={workspaceClassName}>
        <section className="glass-panel workspace-panel workspace-panel--results">
          <div className="section-header">
            <div>
              <div className="section-kicker">展示路线</div>
              <div className="section-title">
                <Sparkle size={20} weight="fill" color="var(--accent-cyan)" />
                推荐路线预览
              </div>

            </div>
            <div className="results-pill">{resultHint}</div>
          </div>

          <div className="results-content">
            <div className="route-grid">
              {latestRoutes.map((route) => (
                <button
                  type="button"
                  className="route-card route-card--compact"
                  key={route.id}
                  onClick={() => setSelectedRoute(route)}
                >
                  <div className="route-card-top">
                    <div className="route-badge">{route.badge}</div>
                    <span className="route-card-action">查看详情</span>
                  </div>

                  <div className="route-title-row">
                    <h3>{route.title}</h3>
                    <span className="route-confidence">{route.confidenceLabel}</span>
                  </div>

                  {route.estimatedBudget && (
                    <div className="route-summary-row route-summary-row--compact">
                      <div className="route-budget route-budget--pill">{route.estimatedBudget}</div>
                    </div>
                  )}

                  <div className="route-mini-stops">
                    {getRouteStops(route).map((item) => (
                      <div
                        className={`route-mini-stop ${item.candidate ? '' : 'route-mini-stop--placeholder'}`}
                        key={`${route.id}-${item.key}`}
                      >
                        <div className="route-mini-stop-label">
                          {getPoiIcon(item.key)}
                          <span>{item.label}</span>
                        </div>
                        <div className="route-mini-stop-name">
                          {item.candidate ? item.candidate.name : `待补齐${item.label}`}
                        </div>
                      </div>
                    ))}
                  </div>
                </button>
              ))}

              {latestRoutes.length === 0 && stage !== 'gathering' && (
                <div className={`results-empty ${isInputFocus ? 'results-empty--secondary' : ''}`}>
                  <div className="results-empty-title">路线预览</div>
                  <p className="results-empty-copy">
                    3 张路线卡，包含景点、餐饮和酒店。
                  </p>
                </div>
              )}

              {stage === 'gathering' && (
                <div className="results-loading">
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <p>正在组合 3 条路线，请稍候...</p>
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="glass-panel workspace-panel workspace-panel--conversation">
          <div className={`summary-card ${isInputFocus ? 'summary-card--hero' : ''}`}>
            <div className="summary-kicker">{isInputFocus ? '开始使用' : '本次需求'}</div>
            <div className="summary-title">
              {isInputFocus ? '描述您的需求，我们来聚合候选方案。' : (latestUserMessage || '本次需求已生成摘要')}
            </div>
            <p className="summary-copy">
              {isInputFocus ? '支持直接输入文本，也可以上传语音。建议尽量描述城市、人数、预算和偏好。' : (latestTags ? buildAnalysisSummary(latestTags) : '本次需求摘要会显示在这里。')}
            </p>

            {!isInputFocus && latestTags && (
              <div className="summary-grid">
                {latestTags.destination && (
                  <div className="summary-chip">
                    <span className="summary-chip-label">目的地</span>
                    <span>{latestTags.destination}</span>
                  </div>
                )}
                {latestTags.people_count && (
                  <div className="summary-chip">
                    <span className="summary-chip-label">人数</span>
                    <span>{latestTags.people_count} 人</span>
                  </div>
                )}
                {formatBudget(latestTags) && (
                  <div className="summary-chip">
                    <span className="summary-chip-label">预算</span>
                    <span>{formatBudget(latestTags)}</span>
                  </div>
                )}
                {latestTags.dates && latestTags.dates.length > 0 && (
                  <div className="summary-chip">
                    <span className="summary-chip-label">时间</span>
                    <span>{latestTags.dates.join(' / ')}</span>
                  </div>
                )}
                {latestTags.food_keywords && latestTags.food_keywords.length > 0 && (
                  <div className="summary-chip">
                    <span className="summary-chip-label">美食偏好</span>
                    <span>{latestTags.food_keywords.join(' / ')}</span>
                  </div>
                )}
                {latestTags.hotel_keywords && latestTags.hotel_keywords.length > 0 && (
                  <div className="summary-chip">
                    <span className="summary-chip-label">住宿偏好</span>
                    <span>{latestTags.hotel_keywords.join(' / ')}</span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="stream-header">最近对话</div>

          <div className="stream-feed">
            {visibleMessages.length === 0 && (
              <div className="composer-empty-state">
                <NavigationArrow size={48} weight="thin" style={{ marginBottom: '1rem' }} />
                <p>请描述您的出行需求。</p>
                <p>例如：这周末去天津两天，预算 3000，想吃海鲜，住方便一点的酒店。</p>
              </div>
            )}

            <AnimatePresence>
              {visibleMessages.map((message) => (
                <motion.div
                  key={message.id}
                  className={`message ${message.type}`}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <div className={`message-icon ${message.type}`}>
                    {message.type === 'user' ? <Target size={20} /> : <Sparkle size={20} />}
                  </div>

                  <div className="message-content">
                    <div className="message-body">
                      {message.content && <div className="message-text">{message.content}</div>}

                      {message.tags && (
                        <div className="tags-container">
                          {message.tags.destination && (
                            <div className="tag">
                              <MapPin size={14} />
                              <span className="tag-label">目的地</span>
                              {message.tags.destination}
                            </div>
                          )}
                          {message.tags.people_count && (
                            <div className="tag">
                              <span className="tag-label">人数</span>
                              {message.tags.people_count} 人
                            </div>
                          )}
                          {message.tags.food_keywords?.map((keyword) => (
                            <div className="tag" key={keyword}>
                              <span className="tag-label">美食</span>
                              {keyword}
                            </div>
                          ))}
                          {message.tags.spot_keywords?.map((keyword) => (
                            <div className="tag" key={keyword}>
                              <span className="tag-label">景点</span>
                              {keyword}
                            </div>
                          ))}
                        </div>
                      )}

                      {(message.stage === 'asr_loading' || message.stage === 'extracting' || message.stage === 'gathering') && (
                        <div className="message-status">
                          <div className="loading-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                          </div>
                          <span>
                            {message.stage === 'asr_loading' && '正在识别语音内容...'}
                            {message.stage === 'extracting' && '正在提取关键需求...'}
                            {message.stage === 'gathering' && '正在生成路线...'}
                          </span>
                        </div>
                      )}

                      {message.stage === 'error' && (
                        <div className="message-feedback message-feedback--error">
                          <WarningCircle size={18} />
                          <span>流程失败：{errorDetails}</span>
                        </div>
                      )}

                      {message.stage === 'done' && (
                        <div className="message-feedback message-feedback--success">
                          <CheckCircle size={18} />
                          <span>三条路线已生成。</span>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={endOfMessagesRef} />
          </div>

          <div className="chat-input-wrapper">
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              style={{ display: 'none' }}
              onChange={handleAudioUpload}
            />
            <button
              className="action-btn upload-btn"
              onClick={() => fileInputRef.current?.click()}
              disabled={!['idle', 'done', 'error'].includes(stage)}
              title="上传音频"
            >
              <Microphone
                size={24}
                className={stage === 'asr_loading' ? 'mic-active' : ''}
                weight={stage === 'asr_loading' ? 'fill' : 'regular'}
              />
            </button>

            <textarea
              ref={textAreaRef}
              className="chat-input"
              placeholder="例如，这周末想去天津玩两三天..."
              value={inputText}
              rows={1}
              disabled={!['idle', 'done', 'error'].includes(stage)}
              onChange={(event) => {
                setInputText(event.target.value);
                syncComposerHeight();
              }}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  handleSendText();
                }
              }}
            />

            <button
              className="action-btn primary"
              onClick={handleSendText}
              disabled={!inputText.trim() || !['idle', 'done', 'error'].includes(stage)}
            >
              <PaperPlaneRight size={20} weight="bold" />
            </button>
          </div>
        </section>
      </main>

      <AnimatePresence>
        {selectedRoute && (
          <motion.div
            className="route-modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedRoute(null)}
          >
            <motion.div
              className="route-modal"
              initial={{ opacity: 0, y: 24, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 12, scale: 0.98 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              onClick={(event) => event.stopPropagation()}
              role="dialog"
              aria-modal="true"
              aria-labelledby="route-modal-title"
            >
              <div className="route-modal-header">
                <div>
                  <div className="route-badge">{selectedRoute.badge}</div>
                  <div className="route-title-row route-title-row--modal">
                    <h3 id="route-modal-title">{selectedRoute.title}</h3>
                    <span className="route-confidence">{selectedRoute.confidenceLabel}</span>
                  </div>
                  <p className="route-subtitle route-subtitle--modal">{selectedRoute.subtitle}</p>
                </div>

                <button
                  type="button"
                  className="route-modal-close"
                  onClick={() => setSelectedRoute(null)}
                  aria-label="关闭路线详情"
                >
                  <X size={20} />
                </button>
              </div>

              <div className="route-modal-body">
                {selectedRoute.estimatedBudget && (
                  <div className="route-modal-overview">
                    <div className="route-budget route-budget--pill">{selectedRoute.estimatedBudget}</div>
                  </div>
                )}

                <div className="route-detail-grid">
                  {getRouteStops(selectedRoute).map((item) => (
                    <section
                      className={`route-step route-step--detail ${item.candidate ? '' : 'route-step--placeholder'}`}
                      key={`${selectedRoute.id}-detail-${item.key}`}
                    >
                      <div className="route-step-top">
                        <div className="route-step-label">
                          {getPoiIcon(item.key)}
                          <span>{item.label}</span>
                        </div>
                        {item.candidate?.rating != null && <span className="route-step-rating">★ {item.candidate.rating}</span>}
                      </div>

                      <div className="route-step-name">
                        {item.candidate ? item.candidate.name : `待接入 ${item.label} 节点`}
                      </div>

                      <div className="route-step-address">
                        {item.candidate?.address || '当前链路暂无该类型候选，先保留展示位。'}
                      </div>

                      <div className="route-step-meta">
                        {item.candidate && formatDistance(item.candidate.center_distance_m) && (
                          <span>{formatDistance(item.candidate.center_distance_m)}</span>
                        )}
                        {item.candidate && formatPrice(item.candidate.price_value_cny) && (
                          <span>¥ {formatPrice(item.candidate.price_value_cny)}</span>
                        )}
                        {item.candidate?.final_score != null && (
                          <span>匹配度 {Math.round(item.candidate.final_score * 100)}</span>
                        )}
                      </div>

                      <div className="tags-container tags-container--compact">
                        {item.candidate
                          ? getVisibleCandidateTags(item.candidate, 4).map((tag) => (
                              <div className="tag tag--candidate" key={`${selectedRoute.id}-${item.key}-${tag}`}>
                                {tag}
                              </div>
                            ))
                          : (
                              <div className="tag tag--candidate tag--muted">前端占位展示</div>
                            )}
                      </div>
                    </section>
                  ))}
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
