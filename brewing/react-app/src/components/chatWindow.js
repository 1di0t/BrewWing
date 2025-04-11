import React from "react";
import "./components/chatwindow.css";

function ChatWindow({ messages }) {
  console.log(messages);
  return (
    <div className="chat-window">
      {messages.map((msg, index) => (
        <div
          key={index}
          className={`chat-bubble ${msg.sender === "user" ? "user-bubble" : "bot-bubble"}`}
        >
          <div className="chat-text">
            {msg.text}
          </div>
          <div className={`chat-timestamp ${msg.sender === "user" ? "user-timestamp" : "bot-timestamp"}`}>
            {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </div>
        </div>
      ))}
    </div>
  );
}

export default ChatWindow;
