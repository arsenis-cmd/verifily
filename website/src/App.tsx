import { useEffect, useRef } from 'react';
import Navbar from './sections/Navbar';
import Hero from './sections/Hero';
import ProblemStatement from './sections/ProblemStatement';
import Solution from './sections/Solution';
import Stats from './sections/Stats';
import HowItWorks from './sections/HowItWorks';
import Features from './sections/Features';
import Testimonials from './sections/Testimonials';
import Footer from './sections/Footer';

function App() {
  const mainRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Register ScrollTrigger
    if (window.gsap && window.ScrollTrigger) {
      window.gsap.registerPlugin(window.ScrollTrigger);

      // Smooth scroll behavior for anchor links
      document.querySelectorAll('a[href^="#"]').forEach((anchor: Element) => {
        anchor.addEventListener('click', (e: Event) => {
          e.preventDefault();
          const href = (anchor as HTMLAnchorElement).getAttribute('href');
          if (href) {
            const target = document.querySelector(href);
            if (target) {
              target.scrollIntoView({ behavior: 'smooth' });
            }
          }
        });
      });

      // Refresh ScrollTrigger after all content is loaded
      window.ScrollTrigger.refresh();

      // Handle resize to refresh ScrollTrigger
      const handleResize = () => {
        window.ScrollTrigger.refresh();
      };

      window.addEventListener('resize', handleResize);

      // Final refresh after a short delay to ensure everything is rendered
      const refreshTimeout = setTimeout(() => {
        window.ScrollTrigger.refresh();
      }, 100);

      return () => {
        window.removeEventListener('resize', handleResize);
        clearTimeout(refreshTimeout);
      };
    }
  }, []);

  return (
    <div ref={mainRef} className="relative overflow-x-hidden">
      <Navbar />
      <main>
        <Hero />
        <ProblemStatement />
        <Solution />
        <Stats />
        <HowItWorks />
        <Features />
        <Testimonials />
        <Footer />
      </main>
    </div>
  );
}

export default App;
