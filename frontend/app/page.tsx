'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  id: number;
  content: string;
  isUser: boolean;
  timestamp: Date;
  isTemporary?: boolean;
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
