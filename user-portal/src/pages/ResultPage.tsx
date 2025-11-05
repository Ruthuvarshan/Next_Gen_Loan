/**
 * Result Page
 * Displays prediction results with approval/denial and explanations
 */

import React from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Typography,
  Button,
  Alert,
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
} from '@mui/material';
import { CheckCircle, Cancel, Home, Add } from '@mui/icons-material';
import { PredictionResponse } from '../services/api';

const ResultPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams();
  
  const result: PredictionResponse | undefined = location.state?.result;

  if (!result) {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        <Alert severity="error">
          No prediction result found. Please submit an application first.
        </Alert>
        <Button onClick={() => navigate('/')} sx={{ mt: 2 }}>
          Return to Dashboard
        </Button>
      </Container>
    );
  }

  const isApproved = result.decision.toLowerCase() === 'approved';

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Box textAlign="center" mb={4}>
          {isApproved ? (
            <CheckCircle sx={{ fontSize: 100, color: 'success.main', mb: 2 }} />
          ) : (
            <Cancel sx={{ fontSize: 100, color: 'error.main', mb: 2 }} />
          )}

          <Typography variant="h3" gutterBottom fontWeight="bold">
            {isApproved ? 'Application Approved!' : 'Application Denied'}
          </Typography>

          <Typography variant="body1" color="textSecondary" paragraph>
            Application ID: <strong>{result.applicant_id}</strong>
          </Typography>
        </Box>

        <Divider sx={{ my: 3 }} />

        <Box mb={3}>
          <Typography variant="h6" gutterBottom>
            Prediction Details
          </Typography>

          <Box display="flex" justifyContent="space-between" mt={2}>
            <Box>
              <Typography variant="body2" color="textSecondary">
                Risk Probability
              </Typography>
              <Typography variant="h5">{(result.probability * 100).toFixed(2)}%</Typography>
            </Box>

            <Box>
              <Typography variant="body2" color="textSecondary">
                Confidence
              </Typography>
              <Chip
                label={result.confidence}
                color={result.confidence === 'High' ? 'success' : 'warning'}
                sx={{ mt: 1 }}
              />
            </Box>
          </Box>
        </Box>

        {!isApproved && result.adverse_action_reasons && result.adverse_action_reasons.length > 0 && (
          <Box>
            <Divider sx={{ my: 3 }} />

            <Alert severity="info" sx={{ mb: 2 }}>
              The following adverse action reasons were identified:
            </Alert>

            <Typography variant="h6" gutterBottom>
              Denial Reasons (ECOA Compliant)
            </Typography>

            <List>
              {result.adverse_action_reasons.map((reason, index) => (
                <ListItem key={index} disableGutters>
                  <ListItemText
                    primary={`${index + 1}. ${reason}`}
                    primaryTypographyProps={{ variant: 'body1' }}
                  />
                </ListItem>
              ))}
            </List>

            <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
              These reasons are provided in compliance with the Equal Credit Opportunity Act (ECOA)
              to explain factors that negatively influenced the decision.
            </Typography>
          </Box>
        )}

        <Box display="flex" gap={2} mt={4}>
          <Button
            variant="outlined"
            startIcon={<Home />}
            onClick={() => navigate('/')}
            fullWidth
          >
            Return to Dashboard
          </Button>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => navigate('/application/new')}
            fullWidth
          >
            New Application
          </Button>
        </Box>
      </Paper>
    </Container>
  );
};

export default ResultPage;
