import { useState, useEffect } from 'react'
import axios from 'axios'
import 'bootstrap/dist/css/bootstrap.min.css'
import 'bootstrap-icons/font/bootstrap-icons.css'
import { Container, Row, Col, Spinner } from 'react-bootstrap'
import Header from './components/Header'
import SearchForm from './components/SearchForm'
import ResultsDisplay from './components/ResultsDisplay'
import './App.css'

function App() {
  const [indexTypes, setIndexTypes] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [searchResults, setSearchResults] = useState(null)
  const [searchParams, setSearchParams] = useState({
    query: '',
    indexType: 'lsi_field_weighted',
    method: 'binary',
    topN: 50,
    rerankingTopK: 10,
    useReranking: true
  })

  // Fetch available index types on component mount
  useEffect(() => {
    const fetchIndexTypes = async () => {
      try {
        setLoading(true)
        const response = await axios.get('/api/index-types')
        setIndexTypes(response.data.index_types)
        
        // Set default index type and method if available
        if (response.data.index_types.length > 0) {
          // Default to field-weighted LSI (usually index 2)
          const defaultIndex = response.data.index_types.find(type => type.id === 'lsi_field_weighted') 
            || response.data.index_types[0]
          const defaultMethod = defaultIndex.methods[0]
          
          setSearchParams(prev => ({
            ...prev,
            indexType: defaultIndex.id,
            method: defaultMethod.id
          }))
        }
      } catch (err) {
        console.error('Error fetching index types:', err)
        setError('Failed to load index types. Please make sure the API server is running.')
      } finally {
        setLoading(false)
      }
    }

    fetchIndexTypes()
  }, [])

  // Handle search form submission
  const handleSearch = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await axios.post('/api/search', searchParams)
      setSearchResults(response.data)
    } catch (err) {
      console.error('Search error:', err)
      setError(err.response?.data?.error || 'An error occurred during the search')
    } finally {
      setLoading(false)
    }
  }

  // Handle search parameter changes
  const handleParamChange = (newParams) => {
    setSearchParams(prev => ({
      ...prev,
      ...newParams
    }))
  }

  return (
    <div className="App">
      <Header />
      <Container fluid className="app-container mt-4">
        <div className="d-flex">
          {/* Search panel with fixed width */}
          <div className="search-panel">
            <div className="search-options-container">
              <SearchForm 
                indexTypes={indexTypes}
                searchParams={searchParams}
                onParamChange={handleParamChange}
                onSearch={handleSearch}
                loading={loading}
              />
            </div>
          </div>
          
          {/* Results panel */}
          <div className="results-panel">
            {error && (
              <div className="alert alert-danger" role="alert">
                {error}
              </div>
            )}
            
            {/* Loading state */}
            {loading ? (
              <div className="results-loading text-center p-5">
                <Spinner animation="border" role="status">
                  <span className="visually-hidden">Loading...</span>
                </Spinner>
                <p className="mt-2">Loading search results...</p>
              </div>
            ) : (
              searchResults ? (
                <ResultsDisplay results={searchResults} />
              ) : null
            )}
          </div>
        </div>
      </Container>
    </div>
  )
}

export default App
