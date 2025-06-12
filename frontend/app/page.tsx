'use client';

import { useState, useRef, useEffect } from 'react';

interface Source {
  chunk_index: number;
  similarity_score: number;
  content_preview: string;
  source_url: string;
}

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  timestamp: Date;
  isTemporary?: boolean;
  sources?: Source[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      content: 'こんにちは！Landverse AI Guideへようこそ。何かご質問はありますか？',
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const question = inputValue.trim();
    
    // 1. ユーザーメッセージを追加
    const userMessage: Message = {
      id: Date.now(),
      content: question,
      isUser: true,
      timestamp: new Date(),
    };

    // 2. 一時的なAIメッセージを追加
    const tempAiMessage: Message = {
      id: Date.now() + 1,
      content: '考え中...',
      isUser: false,
      timestamp: new Date(),
      isTemporary: true,
    };

    setMessages(prev => [...prev, userMessage, tempAiMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // 3. APIリクエスト送信
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // 4. 一時的なメッセージを実際の回答で置き換え
      const finalAiMessage: Message = {
        id: tempAiMessage.id,
        content: data.answer || 'すみません、回答を取得できませんでした。',
        isUser: false,
        timestamp: new Date(),
        sources: data.sources || [],
      };

      setMessages(prev => 
        prev.map(msg => 
          msg.id === tempAiMessage.id ? finalAiMessage : msg
        )
      );
    } catch (error) {
      console.error('API呼び出しエラー:', error);
      
      // エラー時の処理
      const errorMessage: Message = {
        id: tempAiMessage.id,
        content: 'エラーが発生しました。しばらく経ってから再度お試しください。',
        isUser: false,
        timestamp: new Date(),
      };

      setMessages(prev => 
        prev.map(msg => 
          msg.id === tempAiMessage.id ? errorMessage : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e as any);
    }
  };

  const SourcesComponent = ({ sources }: { sources: Source[] }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [showAllSources, setShowAllSources] = useState(false);
    
    if (!sources || sources.length === 0) return null;
    
    // 品質レベルを判定する関数
    const getQualityLevel = (score: number) => {
      if (score >= 0.7) return { level: 'high', color: 'text-green-600 dark:text-green-400', message: '高い関連性' };
      if (score >= 0.5) return { level: 'medium', color: 'text-blue-600 dark:text-blue-400', message: '中程度の関連性' };
      if (score >= 0.3) return { level: 'low', color: 'text-yellow-600 dark:text-yellow-400', message: '低い関連性' };
      return { level: 'very-low', color: 'text-red-600 dark:text-red-400', message: '関連性が低い可能性があります' };
    };
    
    // 低品質なコンテンツを判定する関数
    const isLowQualityContent = (source: Source) => {
      const content = source.content_preview.toLowerCase();
      
      // 1. 極短いコンテンツ（100文字未満）
      if (content.length < 100) return true;
      
      // 2. Shop/Kafraのみのコンテンツ
      const hasOnlyShopKafra = content.includes('shop💱') && 
                              content.includes('kafra premium service') && 
                              content.length < 200;
      if (hasOnlyShopKafra) return true;
      
      // 3. 絵文字ばかりのコンテンツ
      const emojiRegex = /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu;
      const emojiCount = (content.match(emojiRegex) || []).length;
      const textLength = content.replace(emojiRegex, '').length;
      if (emojiCount > textLength * 0.3) return true; // 絵文字が30%以上
      
      // 4. 意味のない繰り返し
      const uniqueWords = new Set(content.split(/\s+/));
      if (uniqueWords.size < 5) return true; // ユニークな単語が5個未満
      
      return false;
    };
    
    // 重複するコンテンツを除去（content_previewの最初の100文字で判定）
    const uniqueSources = sources.filter((source, index, self) => 
      index === self.findIndex(s => 
        s.content_preview.substring(0, 100) === source.content_preview.substring(0, 100)
      )
    );
    
    // 低品質コンテンツを除去
    const meaningfulSources = uniqueSources.filter(source => !isLowQualityContent(source));
    
    // 類似度でソート（高い順）
    const sortedSources = meaningfulSources.sort((a, b) => b.similarity_score - a.similarity_score);
    
    // 表示用のソースを決定（20%以上に引き上げ）
    const primarySources = sortedSources.filter(source => source.similarity_score >= 0.20);
    const veryLowSources = sortedSources.filter(source => source.similarity_score < 0.20);
    
    const displaySources = showAllSources ? sortedSources : primarySources;
    
    return (
      <div className="mt-3 border-t border-gray-200 dark:border-gray-600 pt-3">
        {displaySources.length > 0 ? (
          <>
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
            >
              <span className="text-sm">{isExpanded ? '▼' : '▶'}</span>
              <span className="ml-2">参考にした情報源 ({displaySources.length}件)</span>
            </button>
            
            {isExpanded && (
              <div className="mt-3 space-y-3">
                {displaySources.map((source, index) => {
                  const quality = getQualityLevel(source.similarity_score);
                  return (
                    <div key={index} className="bg-gray-50 dark:bg-gray-600 p-3 rounded-lg">
                      <div className="font-medium text-gray-700 dark:text-gray-200 text-xs mb-2 flex items-center flex-wrap">
                        <span>ソース {index + 1} (類似度: {(source.similarity_score * 100).toFixed(1)}%)</span>
                        <span className={`ml-2 ${quality.color} text-xs`}>
                          • {quality.message}
                        </span>
                      </div>
                      <div className="text-gray-600 dark:text-gray-300 text-xs leading-relaxed mb-2">
                        {source.content_preview}
                      </div>
                      {source.source_url !== 'Unknown' && (
                        <a 
                          href={source.source_url} 
                          target="_blank" 
                          rel="noopener noreferrer" 
                          className="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 hover:underline text-xs break-all"
                        >
                          {source.source_url}
                        </a>
                      )}
                    </div>
                  );
                })}
                
                {veryLowSources.length > 0 && !showAllSources && (
                  <button
                    onClick={() => setShowAllSources(true)}
                    className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 underline"
                  >
                    + 関連性の低い情報源も表示 ({veryLowSources.length}件)
                  </button>
                )}
                
                {showAllSources && veryLowSources.length > 0 && (
                  <button
                    onClick={() => setShowAllSources(false)}
                    className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 underline"
                  >
                    - 関連性の低い情報源を非表示
                  </button>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="text-xs text-amber-600 dark:text-amber-400 space-y-1">
            <div>⚠️ この質問に関連する具体的な情報が見つかりませんでした</div>
            <div className="text-gray-500 dark:text-gray-400">
              {sources.length > 0 ? 
                `検索された${sources.length}件の情報源はすべて関連性が低いか、品質が不十分でした` :
                'データベースに関連情報が存在しません'
              }
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <header className="flex-shrink-0 bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="max-w-4xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-center text-gray-800 dark:text-white">
            Landverse AI Guide
          </h1>
        </div>
      </header>

      <main className="flex-1 overflow-hidden flex flex-col max-w-4xl mx-auto w-full">
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl px-4 py-3 rounded-2xl shadow-sm ${
                  message.isUser
                    ? 'bg-blue-500 text-white'
                    : 'bg-white dark:bg-gray-700 text-gray-800 dark:text-white border border-gray-200 dark:border-gray-600'
                }`}
              >
                <p className="text-sm leading-relaxed">{message.content}</p>
                {!message.isUser && message.sources && (
                  <SourcesComponent sources={message.sources} />
                )}
                {isMounted && (
                  <time className={`text-xs mt-1 block ${
                    message.isUser ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </time>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </main>

      <footer className="flex-shrink-0 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="メッセージを入力してください..."
              disabled={isLoading}
              className="flex-1 px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-full 
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                         placeholder-gray-500 dark:placeholder-gray-400
                         disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="px-6 py-3 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 dark:disabled:bg-gray-600
                         text-white font-medium rounded-full transition-colors duration-200
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                         disabled:cursor-not-allowed disabled:hover:bg-gray-300 dark:disabled:hover:bg-gray-600"
            >
              送信
            </button>
          </form>
        </div>
      </footer>
    </div>
  );
}
