import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

const rootElement = document.getElementById('root');

if (!rootElement) {
  document.body.innerHTML = '<h1 style="color: red; padding: 20px;">Error: Root element not found!</h1>';
} else {
  try {
    createRoot(rootElement).render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    console.log('✓ React app rendered successfully');
  } catch (error) {
    console.error('❌ Error rendering React app:', error);
    document.body.innerHTML = `<div style="color: red; padding: 20px;"><h1>Render Error</h1><pre>${error}</pre></div>`;
  }
}
