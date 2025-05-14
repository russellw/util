// tailwind.config.js or tailwind.config.ts
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}', // For App Router
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}