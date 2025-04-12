import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import BubbleButton from '../components/BubbleButton';
import BackgroundEffect from '../components/BackgroundEffect';
import { useUser } from '../context/UserContext';
import websocketService from '../services/websocketService';

const SignupPage = () => {
  const [username, setUsername] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const { updateUser } = useUser();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!username.trim()) {
      setError('Please enter a username');
      return;
    }
    
    setIsLoading(true);
    setError('');

    try {
      // Connect to the WebSocket with the username
      await websocketService.connect(username);
      
      // Update the user context
      updateUser({ username });
      
      // Navigate to the age selection page
      navigate('/age');
    } catch (err) {
      console.error('Error connecting to WebSocket:', err);
      setError('Failed to connect to game server. Please try again.');
      setIsLoading(false);
    }
  };

  return (
    <div className="page">
      <BackgroundEffect />
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-effect"
        style={{ width: '100%', maxWidth: '400px', padding: '2rem', borderRadius: '1rem' }}
      >
        <h1 className="gradient-text" style={{ textAlign: 'center', marginBottom: '1.5rem', fontSize: '1.875rem', fontWeight: 'bold' }}>
          Welcome to Recommendations
        </h1>
        
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="username" className="form-label">
              Choose a username
            </label>
            <input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="form-input"
              placeholder="Enter your username"
            />
            {error && <p className="form-error">{error}</p>}
          </div>
          
          <BubbleButton className="button-full">
            {isLoading ? 'Connecting...' : 'Get Started'}
          </BubbleButton>
        </form>
      </motion.div>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 0.6 }}
        transition={{ delay: 0.5, duration: 0.5 }}
        style={{ marginTop: '2rem', textAlign: 'center', color: '#a1a1aa', maxWidth: '350px' }}
      >
        Join our community to get personalized recommendations based on your preferences
      </motion.p>
    </div>
  );
};

export default SignupPage;
