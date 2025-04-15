import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { UserProvider } from './context/UserContext';
import SignupPage from './pages/SignupPage';
import HobbiesPage from './pages/HobbiesPage';
import MainPage from './pages/MainPage';
import './App.css';

// Get the base URL from Vite's environment variables
const BASE_URL = import.meta.env.BASE_URL || '/videogamerecs/';

function App() {
  return (
    <UserProvider>
      <Router basename={BASE_URL}>
        <Routes>
          <Route path="/" element={<SignupPage />} />
          <Route path="/hobbies" element={<HobbiesPage />} />
          <Route path="/main" element={<MainPage />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Router>
    </UserProvider>
  );
}

export default App;