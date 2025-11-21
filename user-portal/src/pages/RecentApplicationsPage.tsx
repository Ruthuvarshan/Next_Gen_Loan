/**
 * Recent Applications Page
 * Shows all submitted applications with details and reasons
 */

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Button,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  IconButton,
  Collapse,
  Alert,
  CircularProgress,
  Stack,
} from '@mui/material';
import {
  ArrowBack,
  KeyboardArrowDown,
  KeyboardArrowUp,
  CheckCircle,
  Cancel,
  Info,
} from '@mui/icons-material';
import { useAuth } from '../context/AuthContext';
import api from '../services/api';

interface Application {
  id: number;
  application_id: string;
  timestamp: string;
  applicant_name: string;
  prediction: string;
  probability: number;
  confidence: number;
  loan_amount: number;
  loan_term: number;
  loan_purpose: string;
  gender: string;
  race: string;
  adverse_action_reasons?: string[];
}

interface DetailedApplication {
  application_id: string;
  timestamp: string;
  submitted_by: string;
  applicant_name: string;
  applicant_age: number;
  applicant_income: number;
  gender: string;
  race: string;
  loan_amount: number;
  loan_term: number;
  loan_purpose: string;
  prediction: string;
  probability: number;
  confidence: number;
  documents_uploaded: string[];
  adverse_action_reasons: string[] | null;
}

const ApplicationRow: React.FC<{ app: Application }> = ({ app }) => {
  const [open, setOpen] = useState(false);
  const [details, setDetails] = useState<DetailedApplication | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchDetails = async () => {
    if (details) {
      setOpen(!open);
      return;
    }

    setLoading(true);
    try {
      const response = await api.get(`/api/admin/prediction_log/${app.application_id}`);
      setDetails(response.data);
      setOpen(true);
    } catch (error) {
      console.error('Failed to fetch details:', error);
    } finally {
      setLoading(false);
    }
  };

  const isApproved = app.prediction.toLowerCase() === 'approved';
  const riskPercentage = (app.probability * 100).toFixed(1);

  return (
    <>
      <TableRow sx={{ '& > *': { borderBottom: 'unset' } }}>
        <TableCell>
          <IconButton size="small" onClick={fetchDetails} disabled={loading}>
            {loading ? <CircularProgress size={20} /> : open ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
          </IconButton>
        </TableCell>
        <TableCell>{app.application_id}</TableCell>
        <TableCell>{app.applicant_name}</TableCell>
        <TableCell>{new Date(app.timestamp).toLocaleString()}</TableCell>
        <TableCell align="right">${app.loan_amount?.toLocaleString()}</TableCell>
        <TableCell>
          <Chip
            icon={isApproved ? <CheckCircle /> : <Cancel />}
            label={app.prediction.toUpperCase()}
            color={isApproved ? 'success' : 'error'}
            size="small"
          />
        </TableCell>
        <TableCell align="right">
          <Chip
            label={`${riskPercentage}%`}
            color={app.probability < 0.3 ? 'success' : app.probability < 0.7 ? 'warning' : 'error'}
            size="small"
            variant="outlined"
          />
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={7}>
          <Collapse in={open} timeout="auto" unmountOnExit>
            <Box sx={{ margin: 2 }}>
              {details && (
                <Paper elevation={0} sx={{ p: 3, bgcolor: 'grey.50' }}>
                  <Typography variant="h6" gutterBottom>
                    Application Details
                  </Typography>

                  <Stack spacing={2}>
                    {/* Applicant Information */}
                    <Box>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Applicant Information
                      </Typography>
                      <Box display="grid" gridTemplateColumns="repeat(2, 1fr)" gap={1}>
                        <Typography variant="body2">
                          <strong>Name:</strong> {details.applicant_name}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Age:</strong> {details.applicant_age}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Annual Income:</strong> ${details.applicant_income?.toLocaleString()}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Gender:</strong> {details.gender || 'Not provided'}
                        </Typography>
                      </Box>
                    </Box>

                    {/* Loan Details */}
                    <Box>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Loan Details
                      </Typography>
                      <Box display="grid" gridTemplateColumns="repeat(2, 1fr)" gap={1}>
                        <Typography variant="body2">
                          <strong>Amount:</strong> ${details.loan_amount.toLocaleString()}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Term:</strong> {details.loan_term} months
                        </Typography>
                        <Typography variant="body2">
                          <strong>Purpose:</strong> {details.loan_purpose}
                        </Typography>
                        <Typography variant="body2">
                          <strong>Monthly Payment:</strong> ${(details.loan_amount / details.loan_term).toFixed(2)}
                        </Typography>
                      </Box>
                    </Box>

                    {/* Decision Information */}
                    <Box>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Decision Information
                      </Typography>
                      <Box display="grid" gridTemplateColumns="repeat(2, 1fr)" gap={1}>
                        <Typography variant="body2">
                          <strong>Decision:</strong>{' '}
                          <Chip
                            label={details.prediction.toUpperCase()}
                            color={isApproved ? 'success' : 'error'}
                            size="small"
                          />
                        </Typography>
                        <Typography variant="body2">
                          <strong>Risk Score:</strong> {(details.probability * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2">
                          <strong>Confidence:</strong> {(details.confidence * 100).toFixed(0)}%
                        </Typography>
                        <Typography variant="body2">
                          <strong>Submitted By:</strong> {details.submitted_by}
                        </Typography>
                      </Box>
                    </Box>

                    {/* Documents */}
                    {details.documents_uploaded && details.documents_uploaded.length > 0 && (
                      <Box>
                        <Typography variant="subtitle2" color="primary" gutterBottom>
                          Documents Uploaded
                        </Typography>
                        <Box display="flex" gap={1} flexWrap="wrap">
                          {details.documents_uploaded.map((doc, idx) => (
                            <Chip key={idx} label={doc} size="small" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    )}

                    {/* Reasons for Decision */}
                    {!isApproved && details.adverse_action_reasons && details.adverse_action_reasons.length > 0 && (
                      <Alert severity="error" icon={<Info />}>
                        <Typography variant="subtitle2" gutterBottom>
                          <strong>Reasons for Denial:</strong>
                        </Typography>
                        <ul style={{ margin: 0, paddingLeft: 20 }}>
                          {details.adverse_action_reasons.map((reason, idx) => (
                            <li key={idx}>
                              <Typography variant="body2">{reason}</Typography>
                            </li>
                          ))}
                        </ul>
                      </Alert>
                    )}

                    {isApproved && (
                      <Alert severity="success" icon={<CheckCircle />}>
                        <Typography variant="body2">
                          <strong>Application Approved:</strong> The applicant meets all credit requirements and
                          demonstrates strong financial stability. Risk score of {(details.probability * 100).toFixed(1)}%
                          is below the approval threshold.
                        </Typography>
                      </Alert>
                    )}
                  </Stack>
                </Paper>
              )}
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
};

const RecentApplicationsPage: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [applications, setApplications] = useState<Application[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchApplications();
  }, []);

  const fetchApplications = async () => {
    try {
      const response = await api.get('/api/admin/prediction_log?limit=50');
      setApplications(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load applications');
    } finally {
      setLoading(false);
    }
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
          Recent Applications
        </Typography>
        <Typography variant="body1">{user?.full_name}</Typography>
      </Box>

      {/* Main Content */}
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Button startIcon={<ArrowBack />} onClick={() => navigate('/')}>
            Back to Dashboard
          </Button>
          <Button variant="contained" onClick={fetchApplications} disabled={loading}>
            Refresh
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {loading ? (
          <Box display="flex" justifyContent="center" py={8}>
            <CircularProgress />
          </Box>
        ) : applications.length === 0 ? (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <Typography variant="h6" color="text.secondary">
              No applications found
            </Typography>
            <Typography variant="body2" color="text.secondary" mt={1}>
              Submit your first application to see it here
            </Typography>
          </Paper>
        ) : (
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow sx={{ bgcolor: 'grey.100' }}>
                  <TableCell width="50px" />
                  <TableCell>Application ID</TableCell>
                  <TableCell>Applicant Name</TableCell>
                  <TableCell>Submitted</TableCell>
                  <TableCell align="right">Loan Amount</TableCell>
                  <TableCell>Decision</TableCell>
                  <TableCell align="right">Risk Score</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {applications.map((app) => (
                  <ApplicationRow key={app.id} app={app} />
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        <Typography variant="caption" color="text.secondary" display="block" mt={2} textAlign="center">
          Showing {applications.length} most recent applications
        </Typography>
      </Container>
    </Box>
  );
};

export default RecentApplicationsPage;
