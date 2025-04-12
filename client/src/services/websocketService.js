// src/services/websocketService.js
class WebSocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.messageQueue = [];
    this.callbacks = {};
  }

  connect(username) {
    return new Promise((resolve, reject) => {
      // Using the WebSocket URL from your Lambda function
      const wsUrl = 'wss://8wv7noht30.execute-api.us-east-1.amazonaws.com/prod/';
      
      this.socket = new WebSocket(wsUrl);
      this.username = username; // Store username immediately
      
      // Connection timeout handling
      const connectionTimeout = setTimeout(() => {
        if (!this.isConnected) {
          console.error('WebSocket connection timed out');
          reject(new Error('WebSocket connection timed out'));
          
          if (this.socket) {
            this.socket.close();
          }
        }
      }, 10000); // 10 second timeout
      
      this.socket.onopen = () => {
        console.log('WebSocket connected successfully');
        this.isConnected = true;
        clearTimeout(connectionTimeout);
        
        // Process any messages that were queued while disconnected
        this._processQueue();
        
        resolve();
      };
      
      this.socket.onclose = (event) => {
        console.log(`WebSocket disconnected: Code ${event.code}, Reason: ${event.reason || 'No reason provided'}`);
        this.isConnected = false;
        clearTimeout(connectionTimeout);
        
        // If not a normal closure, attempt reconnection after delay
        if (event.code !== 1000 && event.code !== 1001) {
          console.log('Abnormal closure, attempting reconnect in 3 seconds...');
          setTimeout(() => {
            if (this.username && !this.isConnected) {
              console.log('Attempting to reconnect WebSocket...');
              this.connect(this.username)
                .then(() => console.log('Reconnected successfully'))
                .catch(err => console.error('Reconnection failed:', err));
            }
          }, 3000);
        }
      };
      
      this.socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        clearTimeout(connectionTimeout);
        reject(error);
      };
      
      // Track message rates to avoid flooding
      let messageCount = 0;
      let lastMessageTime = Date.now();
      
      this.socket.onmessage = (event) => {
        try {
          // Track message rate
          messageCount++;
          const now = Date.now();
          // Reset count every 5 seconds
          if (now - lastMessageTime > 5000) {
            messageCount = 1;
            lastMessageTime = now;
          }
          
          // Throttle excessive logging if getting too many messages
          const shouldLog = messageCount < 20;
          
          // Only try to parse if there's actual data
          if (event.data && event.data.trim() !== '') {
            const data = JSON.parse(event.data);
            
            // Handle recommendations from Lambda
            if (data.recommendations) {
              // Always log recommendations
              console.log('Received game recommendations:', data.recommendations);
              if (this.callbacks.onRecommendations) {
                this.callbacks.onRecommendations(data);
              }
            } 
            // Handle errors from the Lambda function
            else if (data.message && data.message.includes("error")) {
              // Reduce error logging if getting too many
              if (shouldLog) {
                console.warn('Server reported an error:', data.message);
              }
              
              // If getting too many errors, consider reconnecting
              if (messageCount > 10 && data.message.includes("error")) {
                console.warn('Too many errors, consider reconnecting');
              }
            }
            // Log other messages only if not excessive
            else if (shouldLog) {
              console.log('Received WebSocket message:', data);
            }
          } else if (shouldLog) {
            // Only log empty messages if not getting too many
            console.log('Received empty WebSocket message');
          }
        } catch (error) {
          if (messageCount < 5) {  // Only log a few parse errors
            console.error('Error parsing WebSocket message:', error);
          }
        }
      };
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.isConnected = false;
    }
  }

  sendUserData(userData) {
    // Format message to match Lambda's expected structure
    // Make sure action is specifically sendUserData as expected by Lambda
    const action = userData.action || 'sendUserData';
    
    // Create a clean message object with required fields
    const message = {
      action: action,
      username: userData.username || this.username || 'anonymous',
      age: userData.age || 25,
      games: userData.games || userData.hobbies || []
    };
    
    // Add any additional fields that are safe to include
    // Avoid spreading the entire userData to prevent duplicate/conflicting fields
    if (userData.preferences) message.preferences = userData.preferences;
    if (userData.timestamp) message.timestamp = userData.timestamp;
    
    console.log('Sending user data to WebSocket:', message);
    
    // Ensure the connection is active before sending
    if (this.isConnected) {
      this._sendMessage(message);
    } else {
      console.warn('WebSocket not connected, attempting to queue message');
      this.messageQueue.push(message);
      
      // If we have a username, try to reconnect
      if (this.username) {
        this.connect(this.username)
          .then(() => console.log('Reconnected to WebSocket'))
          .catch(err => console.error('Failed to reconnect:', err));
      }
    }
  }
  
  onRecommendations(callback) {
    this.callbacks.onRecommendations = callback;
  }
  
  _sendMessage(message) {
    // Add a timestamp to the message to identify when it was created
    const messageWithTimestamp = {
      ...message,
      _timestamp: Date.now(),
      _retryCount: message._retryCount || 0
    };
    
    // Only retry messages for up to 30 seconds
    const isTooOld = messageWithTimestamp._timestamp < (Date.now() - 30000);
    const tooManyRetries = messageWithTimestamp._retryCount > 3;
    
    // Skip messages that have been retried too many times or are too old
    if (isTooOld || tooManyRetries) {
      console.warn(`Dropping message: ${isTooOld ? 'too old' : 'too many retries'}`);
      return;
    }
    
    if (this.isConnected && this.socket && this.socket.readyState === WebSocket.OPEN) {
      try {
        // Clone the message and remove internal properties
        const cleanMessage = { ...messageWithTimestamp };
        delete cleanMessage._timestamp;
        delete cleanMessage._retryCount;
        
        // Format message as JSON and send
        const messageStr = JSON.stringify(cleanMessage);
        this.socket.send(messageStr);
        
        // Only log successful sends if not retrying too frequently
        if (messageWithTimestamp._retryCount === 0) {
          console.log('Message sent successfully');
        }
      } catch (error) {
        console.error('Error sending message:', error);
        
        // Increment retry count and queue
        messageWithTimestamp._retryCount += 1;
        this.messageQueue.push(messageWithTimestamp);
      }
    } else {
      // Only log once when initially queuing
      if (messageWithTimestamp._retryCount === 0) {
        console.warn('WebSocket not ready, queuing message');
      }
      
      // Queue the message to send when connected
      this.messageQueue.push(messageWithTimestamp);
    }
  }
  
  _processQueue() {
    const queueLength = this.messageQueue.length;
    if (queueLength === 0) {
      return; // Nothing to process
    }
    
    // Only log if there are actually messages to process
    console.log(`Processing message queue (${queueLength} items)...`);
    
    // Filter out any stale messages before processing
    const now = Date.now();
    const validMessages = this.messageQueue.filter(msg => {
      const isTooOld = msg._timestamp && msg._timestamp < (now - 30000);
      const tooManyRetries = msg._retryCount && msg._retryCount > 3;
      return !isTooOld && !tooManyRetries;
    });
    
    // Replace the queue with filtered messages
    this.messageQueue = validMessages;
    
    // If the WebSocket isn't connected, don't try to send
    if (!this.isConnected || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected, keeping messages in queue for later');
      return;
    }
    
    // Process each message with built-in rate limiting
    let messagesSent = 0;
    const processMessages = () => {
      // Stop if we've processed all messages or connection is gone
      if (this.messageQueue.length === 0 || !this.isConnected) {
        return;
      }
      
      // Get the next message
      const message = this.messageQueue.shift();
      
      // Send using the main send function (which handles retries)
      this._sendMessage(message);
      messagesSent++;
      
      // Process more messages with delay if there are any left
      if (this.messageQueue.length > 0) {
        // Use increasing delays as we send more messages to avoid flooding
        const delay = Math.min(100 * messagesSent, 1000);
        setTimeout(processMessages, delay);
      }
    };
    
    // Start processing
    processMessages();
  }
}

export default new WebSocketService();