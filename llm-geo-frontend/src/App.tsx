import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

type Message = {
  sender: 'user' | 'bot';
  text: string;
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post('http://localhost:8080/process', {
        task: input,
        task_name: 'chat_task',
      });

      const botMessage: Message = {
        sender: 'bot',
        text: res.data.message || '✅ Task completed successfully.',
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err: any) {
      const errorMessage: Message = {
        sender: 'bot',
        text: err?.response?.data?.message || '❌ Something went wrong.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') sendMessage();
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="min-h-screen bg-gray-200 flex items-center justify-center">
      <div className="w-full max-w-md bg-white shadow-2xl rounded-xl flex flex-col" style={{ height: '600px' }}>
        <div className="bg-blue-600 text-white text-xl font-bold px-6 py-4 rounded-t-xl">
          🌍 GeoChat Assistant
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] px-4 py-2 rounded-2xl text-sm ${
                  msg.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-sm'
                    : 'bg-gray-200 text-gray-800 rounded-bl-sm'
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 text-gray-700 px-4 py-2 rounded-2xl text-sm">
                GeoGPT is thinking...
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="p-4 border-t flex gap-2">
          <input
            type="text"
            placeholder="Ask a spatial question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 border border-gray-300 rounded-full px-4 py-2 text-sm focus:outline-none"
          />
          <button
            onClick={sendMessage}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 text-sm rounded-full disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;
