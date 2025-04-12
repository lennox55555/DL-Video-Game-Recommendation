import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import BackgroundEffect from '../components/BackgroundEffect';
import AgeSlider from '../components/AgeSlider';
import BubbleButton from '../components/BubbleButton';
import { useUser } from '../context/UserContext';

const AgePage = () => {
  const { user, updateUser, updateBackgroundSettings } = useUser();
  const navigate = useNavigate();

  // Redirect if no username set
  useEffect(() => {
    if (!user.username) {
      navigate('/');
    }
  }, [user.username, navigate]);

  const handleAgeChange = (age) => {
    updateUser({ age });
    
    // Update background effects based on age
    const brightness = 80 + (age - 13) * 0.5; // Increases brightness with age
    const glow = Math.min(20, (age - 13) * 0.3); // Increases glow with age, up to a max
    
    updateBackgroundSettings({ brightness, glow });
  };

  const handleContinue = () => {
    navigate('/hobbies');
  };

  return (
    <div className="page">
      <BackgroundEffect />
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="glass-effect"
        style={{ width: '100%', maxWidth: '400px', padding: '2rem' }}
      >
        <h1 className="gradient-text" style={{ textAlign: 'center', marginBottom: '0.5rem' }}>
          PLAYER AGE
        </h1>
        
        <p style={{ textAlign: 'center', color: 'var(--color-accent-1)', marginBottom: '2rem', fontFamily: 'VT323, monospace', fontSize: '1.2rem' }}>
          SET YOUR AGE FOR GAME RECOMMENDATIONS
        </p>
        
        <div style={{ marginBottom: '2.5rem' }}>
          <AgeSlider 
            value={user.age} 
            onChange={handleAgeChange} 
          />
        </div>
        
        <BubbleButton onClick={handleContinue} className="button-full">
          NEXT LEVEL
        </BubbleButton>
      </motion.div>
    </div>
  );
};

export default AgePage;
