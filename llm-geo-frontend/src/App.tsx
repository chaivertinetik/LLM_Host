import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

type Message = {
  sender: 'user' | 'bot';
  text: string;
};

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg: Message = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const res = await axios.post('http://localhost:8080/process', {
        task: input,
        task_name: 'chatbot_task',
      });

      const botMsg: Message = {
        sender: 'bot',
        text: res.data.message || 'Task completed.',
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (error: any) {
      const errMsg: Message = {
        sender: 'bot',
        text: error?.response?.data?.message || 'Something went wrong.',
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') sendMessage();
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div id="chatbox">
      <h2>VertinetikGPT</h2>
      <div id="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={msg.sender}>
            {msg.text}
          </div>
        ))}
        {loading && <div className="bot">VertinetikGPT is thinking...</div>}
        <div ref={messagesEndRef} />
      </div>

      <input
        type="text"
        placeholder="Enter your spatial question..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        style={{ width: '75%' }}
      />
      <button onClick={sendMessage} disabled={loading}>
        Send
      </button>S
    </div>
  );
}

export default App;
