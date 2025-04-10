import React from "react";

function Message({ sender, text }) {
  const isUser = sender === "user";
  return (
    <div
      style={{
        ...styles.messageBubble,
        ...(isUser ? styles.userStyle : styles.botStyle),
        alignSelf: isUser ? "flex-end" : "flex-start",
      }}
    >
      <p>{text}</p>
    </div>
  );
}

const styles = {
  messageBubble: {
    maxWidth: "60%",
    margin: "5px",
    padding: "10px",
    borderRadius: "10px",
    color: "#fff",
  },
  userStyle: {
    backgroundColor: "#4a90e2",
  },
  botStyle: {
    backgroundColor: "#7B61FF",
  },
};

export default Message;
