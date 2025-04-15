import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import BackgroundEffect from '../components/BackgroundEffect';
import BubbleButton from '../components/BubbleButton';
import { useUser } from '../context/UserContext';
import websocketService from '../services/websocketService';

const MainPage = () => {
  const { user } = useUser();
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Redirect if no username set
  useEffect(() => {
    if (!user.username) {
      navigate('/');
      return;
    }

    // Set up a listener for WebSocket recommendations from the Lambda function
    websocketService.onRecommendations((data) => {
      console.log('Received recommendations from server:', data.recommendations);
      setRecommendations(data.recommendations || []);
      setLoading(false);
    });

    // If we don't receive recommendations within 5 seconds, show fallback recommendations
    const fallbackTimer = setTimeout(() => {
      if (loading) {
        console.log('Using fallback recommendations due to timeout');
        setRecommendations([
          { id: 1, title: "Fallback Recommendation 1", description: "This is based on your interest in " + (user.hobbies[0] || "your selections") },
          { id: 2, title: "Fallback Recommendation 2", description: "For " + user.age + " year olds like you who enjoy " + (user.hobbies[1] || user.hobbies[0] || "your selections") }
        ]);
        setLoading(false);
      }
    }, 30000);

    // REMOVED the automatic request for recommendations 
    // as they should have already been sent by HobbiesPage.jsx
    console.log('Waiting for recommendations from HobbiesPage request...');

    return () => {
      clearTimeout(fallbackTimer);
    };
  }, [user, navigate, loading]);

  return (
    <div className="page" style={{ alignItems: 'center', justifyContent: 'flex-start', paddingTop: '4rem' }}>
      <BackgroundEffect />
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        style={{ width: '100%', maxWidth: '768px' }}
      >
        <div className="glass-effect" style={{ padding: '2rem', borderRadius: '1rem', marginBottom: '1.5rem' }}>
          <h1 className="gradient-text" style={{ textAlign: 'center', marginBottom: '0.5rem', fontSize: '1.875rem', fontWeight: 'bold' }}>
            Welcome, {user.username}!
          </h1>
          
          <p style={{ textAlign: 'center', color: '#cbd5e1', marginBottom: '1rem' }}>
            Your personalized space is ready
          </p>
          
          <p style={{ textAlign: 'center', color: 'var(--color-accent-1)', marginBottom: '1rem', fontFamily: 'VT323, monospace' }}>
            Using <span style={{ fontWeight: 'bold' }}>{user.modelType || 'Traditional'}</span> recommendation model
          </p>
          
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'center', marginBottom: '1rem' }}>
            {user.hobbies.map((hobby) => (
              <div key={hobby} className="hobby-pill">
                {hobby}
              </div>
            ))}
          </div>
        </div>

        <div className="glass-effect" style={{ padding: '2rem', borderRadius: '1rem' }}>
          <h2 className="gradient-text" style={{ fontSize: '1.5rem', textAlign: 'center', marginBottom: '1.5rem' }}>GAME RECOMMENDATIONS</h2>
          
          {loading ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '2rem 0' }}>
              <div className="loading-spinner" style={{ marginBottom: '1rem' }}></div>
              <p style={{ 
                fontFamily: 'VT323, monospace',
                fontSize: '1.3rem',
                color: 'var(--color-accent-1)'
              }}>
                SCANNING GAME DATABASE...
              </p>
            </div>
          ) : recommendations.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {recommendations.map((rec) => (
                <motion.div
                  key={rec.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3 }}
                  className="glass-effect"
                  style={{ 
                    padding: '1rem', 
                    marginBottom: '1rem',
                    border: '3px solid var(--color-accent-1)',
                    position: 'relative',
                    overflow: 'hidden'
                  }}
                >
                  <div 
                    style={{ 
                      position: 'absolute', 
                      top: 0, 
                      left: 0, 
                      width: '100%', 
                      height: '4px', 
                      background: `linear-gradient(to right, var(--color-primary), var(--color-accent-1))` 
                    }}
                  />
                  <h3 style={{ 
                    fontFamily: 'VT323, monospace',
                    fontSize: '1.5rem', 
                    marginBottom: '0.5rem',
                    color: 'var(--color-accent-1)',
                    textTransform: 'uppercase',
                    textShadow: '2px 2px 0 rgba(0,0,0,0.5)'
                  }}>
                    {rec.title}
                  </h3>
                  <p style={{ 
                    color: 'var(--color-text)',
                    fontFamily: 'VT323, monospace',
                    fontSize: '1.2rem'
                  }}>
                    {rec.description}
                  </p>
                </motion.div>
              ))}
            </div>
          ) : (
            <div style={{ 
              textAlign: 'center', 
              padding: '2rem 0', 
              color: 'var(--color-accent-1)',
              fontFamily: 'VT323, monospace',
              fontSize: '1.5rem'
            }}>
              [ NO RECOMMENDATIONS FOUND ]
            </div>
          )}
          
          {/* Refresh button removed */}
        </div>
      </motion.div>
    </div>
  );
};

export default MainPage;