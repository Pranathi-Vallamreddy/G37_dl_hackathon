/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: {
          primary:   '#0a0c10',
          secondary: '#111318',
          tertiary:  '#181c24',
          card:      '#1e2330',
        },
        border: {
          DEFAULT: '#2a3040',
          strong:  '#3a4560',
        },
        accent: {
          green:  '#00e5b8',
          blue:   '#3b82f6',
          orange: '#fb923c',
        },
        status: {
          danger:  '#ff4455',
          warning: '#ffb300',
          success: '#00e5b8',
        },
      },
      fontFamily: {
        sans: ['Outfit', 'sans-serif'],
        mono: ['DM Mono', 'monospace'],
      },
      animation: {
        'pulse-dot': 'pulse-dot 2s ease-in-out infinite',
        'fade-in':   'fade-in 0.3s ease-out',
        'slide-up':  'slide-up 0.35s ease-out',
      },
      keyframes: {
        'pulse-dot': {
          '0%,100%': { opacity: '1' },
          '50%':     { opacity: '0.3' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to:   { opacity: '1' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
