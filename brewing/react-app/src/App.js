import React, { useState, useEffect } from "react";
import ChatWindow from "./components/chatWindow";
import InputBar from "./components/inputBar";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [csrfToken, setCsrfToken] = useState("");

  const backendUrl = "http://localhost:8080";

  // CSRF 토큰 가져오기
  useEffect(() => {
    const fetchCsrfToken = async () => {
      try {
        const response = await fetch(backendUrl+"/csrf/", {
          credentials: 'include',
        });
        const data = await response.json();
        console.log("CSRF 토큰:", data.csrfToken);
        setCsrfToken(data.csrfToken);
      } catch (error) {
        console.error("CSRF 토큰 가져오기 실패:", error);
      }
    };
    fetchCsrfToken();
  }, []);

  const handleSendMessage = async (userInput) => {
    const newUserMessage = {
      sender: "user",
      text: userInput,
      timestamp: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setLoading(true);

    try {
      console.log("사용자 입력:", userInput);
      console.log("CSRF 토큰:", csrfToken);
      const response = await fetch(backendUrl+"/api/recommend/", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json",
          "X-CSRFToken": document.cookie
          .split('; ')
          .find(row => row.startsWith('csrftoken='))
          ?.split('=')[1] || ''
        },
        credentials: 'include',
        body: JSON.stringify({ query: userInput })
      });

      if (!response.ok) throw new Error('CSRF 토큰 요청 실패');

      // 챗봇 서버에 메시지 전송
      const data = await response.json();

      const { response: serverResponse } = data;

      const newBotMessage = {
        sender: "bot",
        text: serverResponse?.result || "추천 결과를 불러올 수 없습니다",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, newBotMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage = {
        sender: "bot",
        text: "서버 연결에 문제가 발생했습니다. 다시 시도해주세요.",
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1 className="app-title">BrewWing</h1>
      </header>
      <ChatWindow messages={messages} />
      {loading && <div className="loading-message">추천 중입니다 ☕...</div>}
      <InputBar onSendMessage={handleSendMessage} />
    </div>
  );
}

export default App;
