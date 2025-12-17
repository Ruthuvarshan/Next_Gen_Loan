/**
 * Spark Monitor Page
 * Control panel for Spark UI demo and monitoring
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Paper,
  Grid,
  Card,
  CardContent,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  ArrowBack,
  PlayArrow,
  OpenInNew,
  CheckCircle,
  Info,
  Timeline,
  Storage,
  Memory,
  Speed,
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

const SparkMonitorPage: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [demoStatus, setDemoStatus] = useState<'idle' | 'starting' | 'running'>('idle');
  const [message, setMessage] = useState('');

  const startSparkDemo = async () => {
    setDemoStatus('starting');
    setMessage('Starting Spark demo...');
    
    try {
      // Call backend to start Spark demo
      const response = await api.post('/api/spark/start-demo');
      setDemoStatus('running');
      setMessage(response.data.message || 'Spark demo is running! Check Spark UI for activity.');
    } catch (error: any) {
      setDemoStatus('idle');
      setMessage(error.response?.data?.detail || 'Failed to start Spark demo. Make sure the backend is running.');
    }
  };

  const openSparkUI = () => {
    window.open('http://localhost:4040', '_blank');
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'grey.50' }}>
      {/* Top Bar */}
      <Box
        sx={{
          bgcolor: 'primary.main',
          color: 'white',
          p: 2,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Typography variant="h5" fontWeight="bold">
          Spark Monitoring
        </Typography>
        <Typography variant="body1">{user?.full_name}</Typography>
      </Box>

      {/* Main Content */}
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Button startIcon={<ArrowBack />} onClick={() => navigate('/')}>
            Back to Dashboard
          </Button>
        </Box>

        <Typography variant="h4" gutterBottom fontWeight="bold">
          Apache Spark Monitoring Console
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Monitor distributed ML training jobs, stages, and resource usage in real-time.
        </Typography>

        {message && (
          <Alert severity={demoStatus === 'running' ? 'success' : 'info'} sx={{ mb: 3 }}>
            {message}
          </Alert>
        )}

        <Grid container spacing={3}>
          {/* Control Panel */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Demo Control Panel
                </Typography>
                <Divider sx={{ mb: 2 }} />
                
                <Typography variant="body2" color="text.secondary" paragraph>
                  Start the interactive Spark demo to see Jobs, Stages, and Storage in action.
                </Typography>

                <Box display="flex" flexDirection="column" gap={2}>
                  <Button
                    variant="contained"
                    size="large"
                    fullWidth
                    startIcon={<PlayArrow />}
                    onClick={startSparkDemo}
                    disabled={demoStatus !== 'idle'}
                  >
                    {demoStatus === 'idle' ? 'Start Spark Demo' : demoStatus === 'starting' ? 'Starting...' : 'Demo Running'}
                  </Button>

                  <Button
                    variant="outlined"
                    size="large"
                    fullWidth
                    color="warning"
                    startIcon={<OpenInNew />}
                    onClick={openSparkUI}
                  >
                    Open Spark UI
                  </Button>
                </Box>

                <Alert severity="info" sx={{ mt: 2 }} icon={<Info />}>
                  <Typography variant="caption">
                    <strong>Note:</strong> After starting the demo, the Spark UI will automatically populate with jobs and stages. 
                    Open the Spark UI in a new tab to watch real-time activity.
                  </Typography>
                </Alert>
              </CardContent>
            </Card>
          </Grid>

          {/* Spark UI Features */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  What You'll See in Spark UI
                </Typography>
                <Divider sx={{ mb: 2 }} />

                <List dense>
                  <ListItem>
                    <ListItemIcon>
                      <Timeline color="primary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Jobs Tab"
                      secondary="All Spark jobs with execution times and status"
                    />
                  </ListItem>

                  <ListItem>
                    <ListItemIcon>
                      <Speed color="secondary" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Stages Tab"
                      secondary="Task-level execution details and DAG visualization"
                    />
                  </ListItem>

                  <ListItem>
                    <ListItemIcon>
                      <Storage color="success" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Storage Tab"
                      secondary="Cached DataFrames and memory usage"
                    />
                  </ListItem>

                  <ListItem>
                    <ListItemIcon>
                      <Memory color="warning" />
                    </ListItemIcon>
                    <ListItemText
                      primary="Executors Tab"
                      secondary="CPU, memory usage, and active tasks"
                    />
                  </ListItem>
                </List>
              </CardContent>
            </Card>
          </Grid>

          {/* Demo Workflow */}
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Demo Workflow Steps
              </Typography>
              <Divider sx={{ mb: 2 }} />

              <Grid container spacing={2}>
                {[
                  { step: 1, title: 'Create Spark Session', desc: 'Initialize Spark with UI enabled' },
                  { step: 2, title: 'Load Sample Data', desc: 'Load 10,000 loan records' },
                  { step: 3, title: 'Data Preprocessing', desc: 'Handle missing values and transformations' },
                  { step: 4, title: 'Feature Engineering', desc: 'Create ML features from raw data' },
                  { step: 5, title: 'Build ML Pipeline', desc: 'Assemble pipeline with encoders and model' },
                  { step: 6, title: 'Train GBT Model', desc: 'Train Gradient Boosted Trees classifier' },
                  { step: 7, title: 'Evaluate Performance', desc: 'Calculate metrics and predictions' },
                  { step: 8, title: 'Save Model', desc: 'Persist trained model to disk' },
                ].map((item) => (
                  <Grid item xs={12} sm={6} md={3} key={item.step}>
                    <Box display="flex" alignItems="flex-start" gap={1}>
                      <CheckCircle color="success" fontSize="small" />
                      <Box>
                        <Typography variant="body2" fontWeight="bold">
                          {item.step}. {item.title}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {item.desc}
                        </Typography>
                      </Box>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>

          {/* Quick Access */}
          <Grid item xs={12}>
            <Alert severity="warning" icon={<Info />}>
              <Typography variant="body2">
                <strong>Manual Access:</strong> If the auto-start doesn't work, you can manually run the demo by opening a terminal and executing: 
                <code style={{ marginLeft: 8, padding: '2px 6px', background: '#f5f5f5', borderRadius: 4 }}>
                  cd r:\SSF\Next_Gen_Loan && run_spark_ui_demo.cmd
                </code>
              </Typography>
            </Alert>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
};

export default SparkMonitorPage;
