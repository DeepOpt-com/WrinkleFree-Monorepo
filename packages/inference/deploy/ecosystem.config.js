// PM2 ecosystem configuration for BitNet inference services
// Deploy with: pm2 start ecosystem.config.js

module.exports = {
  apps: [
    {
      name: 'bitnet-native',
      script: '/opt/wrinklefree/packages/inference/deploy/start_bitnet.sh',
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      watch: false,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/pm2/bitnet-native-error.log',
      out_file: '/var/log/pm2/bitnet-native-out.log',
      merge_logs: true
    },
    {
      name: 'streamlit-ui',
      script: '/opt/wrinklefree/packages/inference/deploy/start_streamlit.sh',
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,
      watch: false,
      log_date_format: 'YYYY-MM-DD HH:mm:ss',
      error_file: '/var/log/pm2/streamlit-ui-error.log',
      out_file: '/var/log/pm2/streamlit-ui-out.log',
      merge_logs: true
    }
  ]
};
