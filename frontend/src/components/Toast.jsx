import React from 'react';

export default function Toast({ message, type = 'info', onClose }) {
  if (!message) return null;

  return (
    <div
      className={`toast-notification toast-${type}`}
      style={{
        position: 'fixed',
        bottom: '1.5rem',
        right: '1.5rem',
        zIndex: 9999,
        padding: '0.85rem 1.25rem',
        borderRadius: '8px',
        fontSize: '0.9rem',
        fontWeight: 500,
        boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
        display: 'flex',
        alignItems: 'center',
        gap: '0.75rem',
        background: type === 'error' ? 'var(--red, #DC2626)' : type === 'success' ? 'var(--green, #16A34A)' : 'var(--accent, #0D9488)',
        color: '#FFFFFF',
        animation: 'slideUp 0.25s ease-out',
      }}
    >
      <span>{message}</span>
      <button
        type="button"
        onClick={onClose}
        style={{
          background: 'transparent',
          border: 'none',
          color: '#FFFFFF',
          cursor: 'pointer',
          fontWeight: 700,
          fontSize: '1rem',
          lineHeight: 1,
        }}
      >
        &times;
      </button>
    </div>
  );
}

