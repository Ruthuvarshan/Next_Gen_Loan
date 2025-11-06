/**
 * New Application Page
 * Multi-step wizard for submitting loan applications
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Container,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  TextField,
  Grid,
  MenuItem,
  Alert,
  CircularProgress,
  Card,
  CardContent,
} from '@mui/material';
import { CloudUpload, Home, ExitToApp } from '@mui/icons-material';
import { submitApplication, type PredictionRequest } from '../services/api';
import { useAuth } from '../context/AuthContext';

const steps = ['Applicant Details', 'Loan Details', 'Document Upload', 'Review & Submit'];

const NewApplicationPage: React.FC = () => {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Form data
  const [formData, setFormData] = useState<PredictionRequest>({
    applicant_name: '',
    credit_score: 650,
    age: 30,
    loan_amount: 10000,
    loan_term: 36,
    annual_income: undefined,
    loan_purpose: '',
    sex: '',
    race: '',
  });

  // Files
  const [paystub, setPaystub] = useState<File | null>(null);
  const [bankStatement, setBankStatement] = useState<File | null>(null);

  const handleNext = () => {
    setActiveStep((prev) => prev + 1);
    setError('');
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleChange = (field: keyof PredictionRequest) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData({ ...formData, [field]: e.target.value });
  };

  const handleFileChange = (type: 'paystub' | 'bankStatement') => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    const file = e.target.files?.[0];
    if (file) {
      if (type === 'paystub') setPaystub(file);
      else setBankStatement(file);
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError('');

    try {
      const result = await submitApplication(formData, paystub || undefined, bankStatement || undefined);
      
      // Navigate to results page with the result
      navigate(`/result/${result.applicant_id}`, { state: { result } });
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit application');
      setLoading(false);
    }
  };

  // Validation functions
  const isStep0Valid = () => {
    return (
      formData.applicant_name.length > 0 &&
      formData.credit_score >= 300 &&
      formData.credit_score <= 850 &&
      formData.age >= 18
    );
  };

  const isStep1Valid = () => {
    return formData.loan_amount > 0 && formData.loan_term > 0;
  };

  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Applicant Name"
                value={formData.applicant_name}
                onChange={handleChange('applicant_name')}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Credit Score"
                type="number"
                value={formData.credit_score}
                onChange={handleChange('credit_score')}
                InputProps={{ inputProps: { min: 300, max: 850 } }}
                required
                helperText="300-850"
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Age"
                type="number"
                value={formData.age}
                onChange={handleChange('age')}
                InputProps={{ inputProps: { min: 18, max: 100 } }}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Gender (Optional)"
                value={formData.sex || ''}
                onChange={handleChange('sex')}
              >
                <MenuItem value="">Prefer not to say</MenuItem>
                <MenuItem value="Male">Male</MenuItem>
                <MenuItem value="Female">Female</MenuItem>
                <MenuItem value="Other">Other</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                select
                label="Race (Optional)"
                value={formData.race || ''}
                onChange={handleChange('race')}
              >
                <MenuItem value="">Prefer not to say</MenuItem>
                <MenuItem value="White">White</MenuItem>
                <MenuItem value="Black">Black or African American</MenuItem>
                <MenuItem value="Asian">Asian</MenuItem>
                <MenuItem value="Hispanic">Hispanic or Latino</MenuItem>
                <MenuItem value="Other">Other</MenuItem>
              </TextField>
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Loan Amount"
                type="number"
                value={formData.loan_amount}
                onChange={handleChange('loan_amount')}
                InputProps={{ inputProps: { min: 1000, step: 1000 } }}
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                fullWidth
                label="Loan Term (months)"
                type="number"
                value={formData.loan_term}
                onChange={handleChange('loan_term')}
                InputProps={{ inputProps: { min: 12, max: 60, step: 6 } }}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Annual Income (Optional)"
                type="number"
                value={formData.annual_income || ''}
                onChange={handleChange('annual_income')}
                InputProps={{ inputProps: { min: 0 } }}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                select
                label="Loan Purpose"
                value={formData.loan_purpose || ''}
                onChange={handleChange('loan_purpose')}
              >
                <MenuItem value="">Select purpose</MenuItem>
                <MenuItem value="debt_consolidation">Debt Consolidation</MenuItem>
                <MenuItem value="home_improvement">Home Improvement</MenuItem>
                <MenuItem value="business">Business</MenuItem>
                <MenuItem value="education">Education</MenuItem>
                <MenuItem value="other">Other</MenuItem>
              </TextField>
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Paystub Upload
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Upload your most recent paystub (PDF format)
                  </Typography>
                  <Button
                    variant="contained"
                    component="label"
                    startIcon={<CloudUpload />}
                    fullWidth
                  >
                    {paystub ? `Selected: ${paystub.name}` : 'Choose File'}
                    <input type="file" accept=".pdf" hidden onChange={handleFileChange('paystub')} />
                  </Button>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Bank Statement Upload
                  </Typography>
                  <Typography variant="body2" color="text.secondary" paragraph>
                    Upload your last 3 months' bank statement (PDF format)
                  </Typography>
                  <Button
                    variant="contained"
                    component="label"
                    startIcon={<CloudUpload />}
                    fullWidth
                  >
                    {bankStatement ? `Selected: ${bankStatement.name}` : 'Choose File'}
                    <input
                      type="file"
                      accept=".pdf"
                      hidden
                      onChange={handleFileChange('bankStatement')}
                    />
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        );

      case 3:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <Typography variant="h6" gutterBottom>
                Review Your Application
              </Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="textSecondary">
                Applicant Name
              </Typography>
              <Typography variant="body1">{formData.applicant_name}</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="textSecondary">
                Credit Score
              </Typography>
              <Typography variant="body1">{formData.credit_score}</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="textSecondary">
                Loan Amount
              </Typography>
              <Typography variant="body1">${formData.loan_amount.toLocaleString()}</Typography>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography variant="body2" color="textSecondary">
                Loan Term
              </Typography>
              <Typography variant="body1">{formData.loan_term} months</Typography>
            </Grid>
            <Grid item xs={12}>
              <Typography variant="body2" color="textSecondary">
                Documents Uploaded
              </Typography>
              <Typography variant="body1">
                {paystub ? '✓ Paystub' : '✗ Paystub'} | {bankStatement ? '✓ Bank Statement' : '✗ Bank Statement'}
              </Typography>
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'grey.50' }}>
      {/* Top Navigation Bar */}
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
          <Button
            variant="outlined"
            color="inherit"
            onClick={() => navigate('/')}
            startIcon={<Home />}
          >
            Dashboard
          </Button>
          <Typography variant="body1">Welcome, {user?.full_name}</Typography>
          <Button variant="outlined" color="inherit" onClick={handleLogout} startIcon={<ExitToApp />}>
            Logout
          </Button>
        </Box>
      </Box>

      {/* Main Content */}
      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Paper elevation={3} sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom align="center">
          New Loan Application
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mt: 3, mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Box sx={{ minHeight: 300 }}>{renderStepContent()}</Box>

        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
          <Button disabled={activeStep === 0 || loading} onClick={handleBack}>
            Back
          </Button>
          <Box>
            {activeStep === steps.length - 1 ? (
              <Button variant="contained" onClick={handleSubmit} disabled={loading}>
                {loading ? <CircularProgress size={24} /> : 'Submit Application'}
              </Button>
            ) : (
              <Button
                variant="contained"
                onClick={handleNext}
                disabled={
                  (activeStep === 0 && !isStep0Valid()) || (activeStep === 1 && !isStep1Valid())
                }
              >
                Next
              </Button>
            )}
          </Box>
        </Box>

        {loading && (
          <Box sx={{ mt: 2 }}>
            <Typography variant="body2" color="textSecondary" align="center">
              Processing application... Analyzing documents... Evaluating risk...
            </Typography>
          </Box>
        )}
      </Paper>
    </Container>
    </Box>
  );
};

export default NewApplicationPage;
