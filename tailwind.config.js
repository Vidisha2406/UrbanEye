/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        pine: {
          DEFAULT: '#22393C',
          dark: '#18292B',
          light: '#2C494D',
        },
        fog: {
          DEFAULT: '#46707E',
          dark: '#355560',
          light: '#5B8696',
        },
        moss: {
          DEFAULT: '#6B8B81',
          dark: '#546F67',
          light: '#83A399',
        },
        clay: {
          DEFAULT: '#AFBB98',
          dark: '#93A17B',
          light: '#C4CFB1',
        },
        cream: {
          DEFAULT: '#CECDB9',
          light: '#E5E4D5',
          soft: '#F4F4EE',
          dark: '#B4B39F',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
      boxShadow: {
        'glass': '0 8px 32px 0 rgba(34, 57, 60, 0.12)',
        'glass-hover': '0 12px 36px 0 rgba(34, 57, 60, 0.20)',
        'subtle': '0 2px 10px 0 rgba(34, 57, 60, 0.06)',
      },
      backdropBlur: {
        'xs': '2px',
      }
    },
  },
  plugins: [],
}
