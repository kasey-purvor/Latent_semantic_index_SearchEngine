import React from 'react';
import { Container, Navbar } from 'react-bootstrap';

const Header = () => {
  return (
    <Navbar bg="dark" variant="dark" expand="lg" className="mb-4">
      <Container>
        <Navbar.Brand href="#">
          <h1 className="fs-3 mb-0">Latent Semantic Search Engine</h1>
        </Navbar.Brand>
        <span className="navbar-text text-light">
          Advanced document retrieval with semantic understanding
        </span>
      </Container>
    </Navbar>
  );
};

export default Header; 