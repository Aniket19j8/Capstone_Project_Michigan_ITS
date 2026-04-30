/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        surface: {
          0: "#0b0b12",
          1: "#111118",
          2: "#18181f",
          3: "#1f1f28",
          4: "#26262f",
        },
        border: "#2a2a35",
        accent: {
          DEFAULT: "#7c3aed",
          hover: "#6d28d9",
          light: "#a78bfa",
          muted: "#4c1d95",
        },
      },
    },
  },
  plugins: [],
};
