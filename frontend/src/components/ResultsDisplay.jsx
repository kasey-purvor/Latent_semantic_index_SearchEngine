import React, { useState } from 'react';
import { Card, Badge, Button, Row, Col, Alert } from 'react-bootstrap';

const ResultItem = ({ result, index }) => {
  const [expanded, setExpanded] = useState(false);
  
  const toggleExpand = () => setExpanded(!expanded);
  
  // Format the abstract for display
  const formatAbstract = (abstract) => {
    if (!abstract) return "No abstract available";
    
    return expanded ? abstract : `${abstract.slice(0, 200)}${abstract.length > 200 ? '...' : ''}`;
  };
  
  return (
    <Card className="mb-3 shadow-sm">
      <Card.Header>
        <div className="d-flex justify-content-between align-items-center">
          <h3 className="fs-5 mb-0">
            {index + 1}. {result.title || 'Untitled Paper'}
          </h3>
          <Badge bg="primary" className="ms-2">
            Score: {result.score.toFixed(4)}
            {result.bert_score && (
              <span className="ms-1">| BERT: {result.bert_score.toFixed(4)}</span>
            )}
          </Badge>
        </div>
      </Card.Header>
      <Card.Body>
        <Card.Text>
          {formatAbstract(result.abstract)}
          {result.abstract && result.abstract.length > 200 && (
            <Button 
              variant="link" 
              size="sm" 
              onClick={toggleExpand} 
              className="p-0 ms-1"
            >
              {expanded ? 'Show less' : 'Show more'}
            </Button>
          )}
        </Card.Text>
        
        {result.topics && result.topics.length > 0 && (
          <div className="mb-2">
            <strong>Topics: </strong>
            {result.topics.slice(0, 5).map((topic, i) => (
              <Badge key={i} bg="secondary" className="me-1">
                {topic}
              </Badge>
            ))}
          </div>
        )}
        
        <Row className="mt-3">
          <Col>
            <small className="text-muted">
              <strong>Paper ID:</strong> {result.paper_id}
              {result.year && (
                <span className="ms-2">
                  <strong>Year:</strong> {result.year}
                </span>
              )}
            </small>
          </Col>
        </Row>
        
        {result.url && (
          <div className="mt-2">
            <a href={result.url} target="_blank" rel="noopener noreferrer">
              View Paper
            </a>
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

const ResultsDisplay = ({ results }) => {
  if (!results) return null;
  
  const { query, indexType, method, totalResults, useReranking, results: searchResults } = results;
  
  // Special message for no results
  if (totalResults === 0) {
    return (
      <Alert variant="info">
        No results found for "{query}". Try a different query or search settings.
      </Alert>
    );
  }
  
  // Index type names for display
  const indexTypeNames = {
    'bm25': 'BM25 baseline',
    'lsi_basic': 'Basic LSI',
    'lsi_field_weighted': 'Field-weighted LSI',
    'lsi_bert_enhanced': 'BERT-enhanced LSI'
  };
  
  // Method names for display
  const methodNames = {
    'binary': 'Binary',
    'tfidf': 'TF-IDF',
    'log_entropy': 'Log-Entropy'
  };
  
  return (
    <div>
      <Card className="mb-4 shadow-sm">
        <Card.Header className="bg-primary text-white">
          <h2 className="fs-5 mb-0">Search Results</h2>
        </Card.Header>
        <Card.Body>
          <p>
            Found <strong>{totalResults}</strong> results for query: <strong>"{query}"</strong>
          </p>
          <p className="mb-0">
            <small>
              Using <strong>{indexTypeNames[indexType] || indexType}</strong> with{' '}
              <strong>{methodNames[method] || method}</strong> query method.
              {useReranking && indexType !== 'bm25' && (
                <span> Results reranked using BERT.</span>
              )}
            </small>
          </p>
        </Card.Body>
      </Card>

      {searchResults.map((result, index) => (
        <ResultItem key={result.paper_id} result={result} index={index} />
      ))}
    </div>
  );
};

export default ResultsDisplay; 