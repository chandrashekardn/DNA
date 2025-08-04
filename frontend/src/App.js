import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './App.css';
import { Button } from './components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Textarea } from './components/ui/textarea';
import { Input } from './components/ui/input';
import { Badge } from './components/ui/badge';
import { Progress } from './components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Alert, AlertDescription } from './components/ui/alert';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [sequence, setSequence] = useState('');
  const [patientId, setPatientId] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [sequenceAnalysis, setSequenceAnalysis] = useState(null);
  const [explainability, setExplainability] = useState(null);
  const [loading, setLoading] = useState(false);
  const [validationErrors, setValidationErrors] = useState([]);
  const [sequenceStats, setSequenceStats] = useState(null);
  const [activeTab, setActiveTab] = useState('input');

  // Debounced sequence validation
  const validateSequence = useCallback((seq) => {
    const errors = [];
    
    if (seq.length > 0 && seq.length < 50) {
      errors.push('Sequence too short. Minimum 50 nucleotides required.');
    }
    
    if (seq.length > 131072) {
      errors.push('Sequence too long. Maximum 131k nucleotides supported.');
    }
    
    const validChars = /^[ATCGN]*$/i;
    if (seq.length > 0 && !validChars.test(seq)) {
      errors.push('Invalid nucleotides found. Only A, T, C, G, N allowed.');
    }
    
    setValidationErrors(errors);
    
    // Calculate sequence statistics
    if (seq.length >= 50 && errors.length === 0) {
      const gcCount = (seq.match(/[GC]/gi) || []).length;
      const stats = {
        length: seq.length,
        gcContent: ((gcCount / seq.length) * 100).toFixed(2),
        composition: {
          A: (seq.match(/A/gi) || []).length,
          T: (seq.match(/T/gi) || []).length,
          C: (seq.match(/C/gi) || []).length,
          G: (seq.match(/G/gi) || []).length,
          N: (seq.match(/N/gi) || []).length
        }
      };
      setSequenceStats(stats);
    } else {
      setSequenceStats(null);
    }
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      if (sequence) {
        validateSequence(sequence);
      } else {
        setValidationErrors([]);
        setSequenceStats(null);
      }
    }, 500);
    
    return () => clearTimeout(timeoutId);
  }, [sequence, validateSequence]);

  const handleSequenceChange = (e) => {
    const value = e.target.value.toUpperCase().replace(/[^ATCGN\n\r\s]/g, '').replace(/[\n\r\s]/g, '');
    setSequence(value);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        let content = event.target.result;
        
        // Parse FASTA format
        if (content.startsWith('>')) {
          const lines = content.split('\n');
          content = lines.slice(1).join('').replace(/\s/g, '');
        }
        
        setSequence(content.toUpperCase());
      };
      reader.readAsText(file);
    }
  };

  const submitAnalysis = async () => {
    if (validationErrors.length > 0 || !sequence) return;
    
    setLoading(true);
    try {
      const response = await axios.post(`${BACKEND_URL}/api/predict`, {
        sequence: sequence,
        patient_id: patientId || null,
        sequence_type: 'genomic',
        max_length: 4096
      });
      
      setPredictions(response.data.predictions);
      setSequenceAnalysis(response.data.sequence_analysis);
      setExplainability(response.data.explainability);
      setActiveTab('results');
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    switch (risk.toLowerCase()) {
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      default: return 'default';
    }
  };

  const getRiskBgColor = (risk) => {
    switch (risk.toLowerCase()) {
      case 'high': return 'bg-red-50 border-red-200';
      case 'medium': return 'bg-yellow-50 border-yellow-200';
      default: return 'bg-green-50 border-green-200';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-600 rounded-lg">
              <div className="h-8 w-8 text-white font-bold text-center">🧬</div>
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Caduceus DNA Analysis</h1>
              <p className="text-gray-600 mt-1">Advanced genomic disease prediction using transformer models</p>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="input" className="flex items-center space-x-2">
              <FileText className="h-4 w-4" />
              <span>Sequence Input</span>
            </TabsTrigger>
            <TabsTrigger value="results" disabled={!predictions}>
              <Activity className="h-4 w-4 mr-2" />
              Disease Predictions
            </TabsTrigger>
            <TabsTrigger value="explainability" disabled={!explainability}>
              <Brain className="h-4 w-4 mr-2" />
              Analysis Details
            </TabsTrigger>
          </TabsList>

          <TabsContent value="input" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <DNA className="h-5 w-5 text-blue-600" />
                  <span>DNA Sequence Input</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Patient ID (Optional)
                  </label>
                  <Input
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                    placeholder="Enter patient identifier"
                    className="mb-4"
                  />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <label className="block text-sm font-medium text-gray-700">
                      DNA Sequence
                    </label>
                    <div className="flex items-center space-x-2">
                      <Input
                        type="file"
                        accept=".fasta,.fa,.txt"
                        onChange={handleFileUpload}
                        className="hidden"
                        id="sequence-file"
                      />
                      <label htmlFor="sequence-file" className="cursor-pointer">
                        <Button variant="outline" size="sm" asChild>
                          <span>
                            <Upload className="h-4 w-4 mr-2" />
                            Upload FASTA
                          </span>
                        </Button>
                      </label>
                    </div>
                  </div>
                  
                  <Textarea
                    value={sequence}
                    onChange={handleSequenceChange}
                    placeholder="Enter DNA sequence (A, T, C, G, N)..."
                    rows={8}
                    className={`font-mono text-sm ${validationErrors.length > 0 ? 'border-red-300 focus:border-red-500' : ''}`}
                  />
                </div>

                {validationErrors.length > 0 && (
                  <Alert variant="destructive">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      <ul className="list-disc list-inside">
                        {validationErrors.map((error, index) => (
                          <li key={index}>{error}</li>
                        ))}
                      </ul>
                    </AlertDescription>
                  </Alert>
                )}

                {sequenceStats && (
                  <Card className="bg-gray-50">
                    <CardHeader>
                      <CardTitle className="text-lg flex items-center space-x-2">
                        <BarChart3 className="h-4 w-4" />
                        <span>Sequence Statistics</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                        <div className="text-center p-2 bg-white rounded border">
                          <div className="text-lg font-bold text-blue-600">{sequenceStats.length}</div>
                          <div className="text-sm text-gray-600">Length (bp)</div>
                        </div>
                        <div className="text-center p-2 bg-white rounded border">
                          <div className="text-lg font-bold text-green-600">{sequenceStats.gcContent}%</div>
                          <div className="text-sm text-gray-600">GC Content</div>
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-4 gap-2 text-center">
                        {Object.entries(sequenceStats.composition).map(([nucleotide, count]) => (
                          <div key={nucleotide} className="p-2 bg-white rounded border">
                            <div className="font-bold text-blue-600">{nucleotide}</div>
                            <div className="text-sm text-gray-600">{count}</div>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}

                <Button
                  onClick={submitAnalysis}
                  disabled={loading || validationErrors.length > 0 || !sequence}
                  size="lg"
                  className="w-full"
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Brain className="h-4 w-4 mr-2" />
                  )}
                  {loading ? 'Analyzing Sequence...' : 'Analyze DNA Sequence'}
                </Button>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="results" className="space-y-6">
            {predictions && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Activity className="h-5 w-5 text-red-600" />
                      <span>Disease Risk Predictions</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid gap-4">
                      {predictions
                        .sort((a, b) => b.probability - a.probability)
                        .map((prediction, index) => (
                        <Card key={index} className={getRiskBgColor(prediction.risk_category)}>
                          <CardContent className="p-4">
                            <div className="flex items-center justify-between mb-3">
                              <h3 className="font-semibold text-lg">{prediction.disease_name}</h3>
                              <Badge variant={getRiskColor(prediction.risk_category)}>
                                {prediction.risk_category} Risk
                              </Badge>
                            </div>
                            
                            <div className="space-y-2">
                              <div className="flex justify-between text-sm">
                                <span>Risk Score:</span>
                                <span className="font-semibold">
                                  {(prediction.probability * 100).toFixed(1)}%
                                </span>
                              </div>
                              
                              <Progress 
                                value={prediction.probability * 100} 
                                className="h-2"
                              />
                              
                              <div className="text-xs text-gray-600">
                                95% CI: {(prediction.confidence_interval[0] * 100).toFixed(1)}% - 
                                {(prediction.confidence_interval[1] * 100).toFixed(1)}%
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {sequenceAnalysis && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Sequence Analysis Summary</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-3 gap-4 text-center">
                        <div>
                          <div className="text-2xl font-bold text-blue-600">{sequenceAnalysis.length}</div>
                          <div className="text-sm text-gray-600">Nucleotides</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-green-600">
                            {(sequenceAnalysis.gc_content * 100).toFixed(1)}%
                          </div>
                          <div className="text-sm text-gray-600">GC Content</div>
                        </div>
                        <div>
                          <div className="text-2xl font-bold text-purple-600">
                            {sequenceAnalysis.complexity_score.toFixed(2)}
                          </div>
                          <div className="text-sm text-gray-600">Complexity</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </TabsContent>

          <TabsContent value="explainability" className="space-y-6">
            {explainability && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Brain className="h-5 w-5 text-purple-600" />
                      <span>Important Genomic Regions</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {explainability.important_regions?.length > 0 ? (
                      <div className="space-y-3">
                        {explainability.important_regions.map((region, index) => (
                          <Card key={index} className="p-4 border-l-4 border-blue-500">
                            <div className="flex justify-between items-start mb-2">
                              <span className="text-sm font-medium">
                                Position {region.start_position}-{region.end_position}
                              </span>
                              <Badge variant={region.effect_type === 'positive' ? 'default' : 'secondary'}>
                                {region.effect_type}
                              </Badge>
                            </div>
                            <div className="text-xs font-mono bg-gray-100 p-2 rounded mb-2">
                              {region.sequence}
                            </div>
                            <div className="text-sm text-gray-600">
                              Impact Score: {region.average_attribution.toFixed(3)}
                            </div>
                          </Card>
                        ))}
                      </div>
                    ) : (
                      <p className="text-gray-600">No significant regions identified in this analysis.</p>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Sequence Motif Analysis</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {explainability.motif_analysis && (
                      <div className="space-y-3">
                        {Object.entries(explainability.motif_analysis).map(([motif, data]) => (
                          <div key={motif} className="flex justify-between items-center p-3 bg-gray-50 rounded">
                            <div>
                              <span className="font-medium">{motif.replace('_', ' ')}</span>
                              <div className="text-sm text-gray-600">
                                {data.occurrences} occurrences
                              </div>
                            </div>
                            <Badge variant="outline">
                              Impact: {data.average_attribution.toFixed(3)}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Analysis Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    {explainability.summary_statistics && (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="p-3 bg-green-50 rounded">
                          <div className="text-lg font-bold text-green-700">
                            {explainability.summary_statistics.total_positive_attribution?.toFixed(3) || '0.000'}
                          </div>
                          <div className="text-sm text-gray-600">Positive Attribution</div>
                        </div>
                        <div className="p-3 bg-red-50 rounded">
                          <div className="text-lg font-bold text-red-700">
                            {Math.abs(explainability.summary_statistics.total_negative_attribution || 0).toFixed(3)}
                          </div>
                          <div className="text-sm text-gray-600">Negative Attribution</div>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-600">
            <p>Caduceus DNA Analysis - Advanced genomic disease prediction</p>
            <p className="text-sm mt-1">Powered by transformer-based DNA sequence modeling</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;