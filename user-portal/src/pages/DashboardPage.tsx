/**
 * Dashboard Page
 * Main landing page for loan officers
 */

import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  CardActions,
} from '@mui/material';
import { Add, Assessment, ExitToApp } from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';

const DashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login');
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
          Loan Officer Portal
        </Typography>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="body1">Welcome, {user?.full_name}</Typography>
          <Button variant="outlined" color="inherit" onClick={handleLogout} startIcon={<ExitToApp />}>
            Logout
          </Button>
        </Box>
      </Box>

      {/* Main Content */}
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" gutterBottom fontWeight="bold">
          Dashboard
        </Typography>
        <Typography variant="body1" color="textSecondary" paragraph>
          Next-Generation Loan Origination System
        </Typography>

        <Grid container spacing={3} mt={2}>
          {/* New Application Card */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Add sx={{ fontSize: 50, color: 'primary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  New Application
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  Start a new loan application with our intelligent document processing and AI-powered
                  risk assessment.
                </Typography>
              </CardContent>
              <CardActions>
                <Button
                  variant="contained"
                  size="large"
                  fullWidth
                  onClick={() => navigate('/application/new')}
                >
                  Start Application
                </Button>
              </CardActions>
            </Card>
          </Grid>

          {/* Recent Applications Card */}
          <Grid item xs={12} md={6}>
            <Card elevation={3}>
              <CardContent>
                <Assessment sx={{ fontSize: 50, color: 'secondary.main', mb: 2 }} />
                <Typography variant="h5" gutterBottom>
                  Recent Applications
                </Typography>
                <Typography variant="body2" color="textSecondary">
                  View and track your recent loan application submissions and their status.
                </Typography>
              </CardContent>
              <CardActions>
                <Button variant="outlined" size="large" fullWidth disabled>
                  Coming Soon
                </Button>
              </CardActions>
            </Card>
          </Grid>
        </Grid>

        {/* System Info */}
        <Box mt={4} p={3} bgcolor="white" borderRadius={2} boxShadow={1}>
          <Typography variant="h6" gutterBottom>
            System Features
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" fontWeight="bold" color="primary">
                ✓ IDP Engine
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Intelligent Document Processing with OCR
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" fontWeight="bold" color="primary">
                ✓ NLP Features
              </Typography>
              <Typography variant="caption" color="textSecondary">
                Advanced bank statement analysis
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" fontWeight="bold" color="primary">
                ✓ XGBoost Model
              </Typography>
              <Typography variant="caption" color="textSecondary">
                State-of-the-art risk prediction
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Typography variant="body2" fontWeight="bold" color="primary">
                ✓ SHAP Explanations
              </Typography>
              <Typography variant="caption" color="textSecondary">
                ECOA-compliant adverse action reasons
              </Typography>
            </Grid>
          </Grid>
        </Box>
      </Container>
    </Box>
  );
};

export default DashboardPage;
