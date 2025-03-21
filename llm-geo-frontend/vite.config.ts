import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  base: '/LLM_Host/', 
  plugins: [react()],
})
