import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { UserProvider } from './context/UserContext';
import SignupPage from './pages/SignupPage';
import AgePage from './pages/AgePage';
import HobbiesPage from './pages/HobbiesPage';
import MainPage from './pages/MainPage';
import './App.css';

function App() {
  return (
    <UserProvider>
      <Router>
        <Routes>
          <Route path="/" element={<SignupPage />} />
          <Route path="/age" element={<AgePage />} />
          <Route path="/hobbies" element={<HobbiesPage />} />
          <Route path="/main" element={<MainPage />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Router>
    </UserProvider>
  );
}

export default App;