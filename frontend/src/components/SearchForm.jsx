import React, { useState, useEffect } from 'react';
import { Form, Button, Card, InputGroup, FormControl, Spinner } from 'react-bootstrap';

const SearchForm = ({ indexTypes, searchParams, onParamChange, onSearch, loading }) => {
  const [localSearchParams, setLocalSearchParams] = useState(searchParams);
  const [supportedMethods, setSupportedMethods] = useState([]);
  const [isBM25, setIsBM25] = useState(false);

  // Update local params when props change
  useEffect(() => {
    setLocalSearchParams(searchParams);
  }, [searchParams]);

  // Update supported methods when index type changes
  useEffect(() => {
    if (indexTypes.length > 0) {
      const selectedIndex = indexTypes.find(type => type.id === localSearchParams.indexType);
      if (selectedIndex) {
        setSupportedMethods(selectedIndex.methods);
        setIsBM25(selectedIndex.id === 'bm25');
        
        // If BM25, force binary method and disable reranking
        if (selectedIndex.id === 'bm25') {
          setLocalSearchParams(prev => ({
            ...prev,
            method: 'binary',
            useReranking: false
          }));
          // Propagate changes to parent
          onParamChange({
            method: 'binary',
            useReranking: false
          });
        }
      }
    }
  }, [indexTypes, localSearchParams.indexType, onParamChange]);

  const handleSubmit = (e) => {
    e.preventDefault();
    onParamChange(localSearchParams);
    onSearch();
  };

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    const newValue = type === 'checkbox' ? checked : value;
    
    setLocalSearchParams({
      ...localSearchParams,
      [name]: newValue
    });
    
    // If it's the index type, don't immediately propagate (wait for other useEffect to handle method changes)
    if (name !== 'indexType') {
      onParamChange({ [name]: newValue });
    } else {
      onParamChange({ indexType: newValue });
    }
  };

  const renderMethodDescription = () => {
    const method = localSearchParams.method;
    
    if (method === 'binary') {
      return "Binary method is best for keyword searches, treating terms as present or absent.";
    } else if (method === 'tfidf') {
      return "TF-IDF weights terms by their frequency in the document and rarity across the corpus. Best for finding entire documents similar to a query.";
    } else if (method === 'log_entropy') {
      return "Log-entropy applies advanced weighting for better handling of varied term importance.";
    }
    
    return "";
  };

  return (
    <Card className="mb-4 shadow-sm">
      <Card.Header className="bg-primary text-white">
        <h2 className="fs-5 mb-0">Search Options</h2>
      </Card.Header>
      <Card.Body>
        <Form onSubmit={handleSubmit}>
          {/* Options Section */}
          <div className="options-section mb-4">
            <div className="row">
              {/* Index Type Selection */}
              <div className="col-12 mb-3">
                <Form.Group controlId="indexType">
                  <Form.Label><strong>Index Type</strong></Form.Label>
                  <Form.Select 
                    name="indexType"
                    value={localSearchParams.indexType}
                    onChange={handleChange}
                    disabled={loading}
                  >
                    {indexTypes.map(type => (
                      <option key={type.id} value={type.id}>
                        {type.name}
                      </option>
                    ))}
                  </Form.Select>
                  <Form.Text className="text-muted">
                    {isBM25 ? 
                      "BM25 is a simple keyword matching baseline." : 
                      "LSI indexes capture semantic relationships between terms."}
                  </Form.Text>
                </Form.Group>
              </div>

              {/* Query Method Selection */}
              <div className="col-md-6 mb-3">
                <Form.Group controlId="method">
                  <Form.Label><strong>Query Method</strong></Form.Label>
                  <Form.Select 
                    name="method"
                    value={localSearchParams.method}
                    onChange={handleChange}
                    disabled={loading || isBM25}
                  >
                    {supportedMethods.map(method => (
                      <option key={method.id} value={method.id}>
                        {method.name}
                      </option>
                    ))}
                  </Form.Select>
                  <Form.Text className="text-muted">
                    {isBM25 ? 
                      "BM25 only supports binary." : 
                      renderMethodDescription()}
                  </Form.Text>
                </Form.Group>
              </div>

              {/* BERT Reranking Toggle */}
              <div className="col-md-6 mb-3">
                {!isBM25 && (
                  <Form.Group controlId="useReranking" className="h-100 d-flex flex-column justify-content-center mt-md-4">
                    <Form.Check 
                      type="checkbox"
                      label="Use BERT reranking"
                      name="useReranking"
                      checked={localSearchParams.useReranking}
                      onChange={handleChange}
                      disabled={loading || isBM25}
                    />
                    <Form.Text className="text-muted">
                      Neural reranking for better relevance
                    </Form.Text>
                  </Form.Group>
                )}
              </div>
            </div>

            <div className="row">
              {/* Results Count */}
              <div className="col-md-6 mb-3">
                <Form.Group controlId="topN">
                  <Form.Label><strong>Initial Results</strong></Form.Label>
                  <Form.Control
                    type="number"
                    name="topN"
                    value={localSearchParams.topN}
                    onChange={handleChange}
                    min={1}
                    max={100}
                    disabled={loading}
                  />
                  <Form.Text className="text-muted">
                    Documents to retrieve initially
                  </Form.Text>
                </Form.Group>
              </div>
              <div className="col-md-6 mb-3">
                <Form.Group controlId="rerankingTopK">
                  <Form.Label><strong>Display Results</strong></Form.Label>
                  <Form.Control
                    type="number"
                    name="rerankingTopK"
                    value={localSearchParams.rerankingTopK}
                    onChange={handleChange}
                    min={1}
                    max={50}
                    disabled={loading}
                  />
                  <Form.Text className="text-muted">
                    Results to display after reranking
                  </Form.Text>
                </Form.Group>
              </div>
            </div>
          </div>

          {/* Search Query - Positioned at the bottom and centered */}
          <div className="search-query-section mt-4 mx-auto" style={{ maxWidth: "90%" }}>
            <Form.Group controlId="query">
              <Form.Label className="text-center w-100"><strong>Search Query</strong></Form.Label>
              <InputGroup>
                <FormControl
                  type="text"
                  placeholder="Enter your search query..."
                  name="query"
                  value={localSearchParams.query}
                  onChange={handleChange}
                  disabled={loading}
                  className="search-input"
                />
                <Button 
                  variant="primary" 
                  type="submit"
                  disabled={loading || !localSearchParams.query.trim()}
                >
                  {loading ? (
                    <>
                      <Spinner
                        as="span"
                        animation="border"
                        size="sm"
                        role="status"
                        aria-hidden="true"
                        className="me-2"
                      />
                      Searching...
                    </>
                  ) : (
                    'Search'
                  )}
                </Button>
              </InputGroup>
              <Form.Text className="text-muted text-center d-block">
                {isBM25 ? 
                  "For BM25, use specific keywords for best results." : 
                  "For LSI, you can use natural language queries or keywords."}
              </Form.Text>
            </Form.Group>
          </div>
        </Form>
      </Card.Body>
    </Card>
  );
};

export default SearchForm; 