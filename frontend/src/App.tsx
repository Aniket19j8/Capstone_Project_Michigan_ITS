import { Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import UserSupportPage from "./pages/UserSupportPage";
import AdminPage from "./pages/AdminPage";
import RagDemoPage from "./pages/RagDemoPage";
import "./index.css";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/support" element={<UserSupportPage />} />
      <Route path="/admin" element={<AdminPage />} />
      <Route path="/rag" element={<RagDemoPage />} />
    </Routes>
  );
}
