import { useState, useEffect } from 'react';
import { ChevronRight } from 'lucide-react';

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav
      className={`fixed top-4 left-1/2 -translate-x-1/2 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-black/90 backdrop-blur-md shadow-lg'
          : 'bg-black/50 backdrop-blur-sm'
      } rounded-full px-6 py-3`}
    >
      <div className="flex items-center gap-8">
        {/* Logo */}
        <a href="#" className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-md logo-gradient flex items-center justify-center">
            <svg viewBox="0 0 24 24" className="w-4 h-4 text-white" fill="currentColor">
              <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/>
            </svg>
          </div>
          <span className="text-white font-semibold text-sm">verifily</span>
        </a>

        {/* Navigation Links */}
        <div className="hidden md:flex items-center gap-6">
          <a href="#how-it-works" className="text-white/80 hover:text-white text-sm transition-colors">
            How it works
          </a>
          <a href="#features" className="text-white/80 hover:text-white text-sm transition-colors">
            Features
          </a>
          <a href="#proof" className="text-white/80 hover:text-white text-sm transition-colors">
            Proof
          </a>
        </div>

        {/* CTA Button */}
        <button className="cta-gradient text-white text-sm font-medium px-4 py-2 rounded-full flex items-center gap-1 hover:opacity-90 transition-opacity">
          Get started
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
