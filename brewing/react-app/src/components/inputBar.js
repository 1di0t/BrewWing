// src/components/InputBar.js
import React, { useState } from "react";
import "./components/inputBar.css";

function InputBar({ onSendMessage }) {
  const [inputValue, setInputValue] = useState("");

  const handleChange = (e) => {
    setInputValue(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter") {
      handleSend();
    }
  };

  const handleSend = () => {
    if (inputValue.trim() === "") return;
    onSendMessage(inputValue);
    setInputValue("");
  };

  return (
    <div className="input-bar">
      <input
        type="text"
        value={inputValue}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="원하시는 커피 취향을 말씀해 주세요"
        className="input-field"
        aria-label="커피 취향 입력"
      />
      <button onClick={handleSend} className="send-button" aria-label="메시지 전송">전송</button>
    </div>
  );
}

export default InputBar;
