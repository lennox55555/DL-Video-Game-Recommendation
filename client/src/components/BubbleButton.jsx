import { motion } from 'framer-motion';

const BubbleButton = ({ onClick, children, primary = true, className = '', disabled = false }) => {
  return (
    <motion.button
      onClick={onClick}
      className={`button ${primary ? 'button-primary' : 'button-secondary'} ${className}`}
      whileHover={{ scale: disabled ? 1 : 1.05 }}
      whileTap={{ scale: disabled ? 1 : 0.95 }}
      transition={{ type: 'spring', stiffness: 400, damping: 17 }}
      disabled={disabled}
      style={{ opacity: disabled ? 0.5 : 1, cursor: disabled ? 'not-allowed' : 'pointer' }}
    >
      {children}
    </motion.button>
  );
};

export default BubbleButton;
