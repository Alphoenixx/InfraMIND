/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-primary': '#050510',
        'bg-secondary': '#0a0a1f',
        'accent-green': '#10b981',
        'accent-blue': '#3b82f6',
        'accent-purple': '#8b5cf6',
        'accent-orange': '#f97316',
        'accent-red': '#ef4444',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      }
    },
  },
  plugins: [],
}
