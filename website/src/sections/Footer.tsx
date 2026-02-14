import { useEffect, useRef } from 'react';

const Footer = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!window.gsap || !window.ScrollTrigger) return;

    const gsap = window.gsap;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        leftRef.current,
        { opacity: 0, x: -50 },
        {
          opacity: 1,
          x: 0,
          duration: 0.9,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: sectionRef.current,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );

      gsap.fromTo(
        rightRef.current,
        { opacity: 0, x: 50 },
        {
          opacity: 1,
          x: 0,
          duration: 0.9,
          delay: 0.15,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: sectionRef.current,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );

      gsap.fromTo(
        bottomRef.current,
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.7,
          delay: 0.3,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: bottomRef.current,
            start: 'top 90%',
            toggleActions: 'play none none reverse',
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <footer
      ref={sectionRef}
      className="relative bg-black py-24 overflow-hidden"
    >
      <div className="max-w-6xl mx-auto px-4">
        <div className="grid md:grid-cols-2 gap-16 mb-16">
          {/* Left — tagline */}
          <div ref={leftRef}>
            <h2 className="text-3xl md:text-5xl font-semibold text-white leading-tight mb-6">
              The missing step
              <br />
              between training
              <br />
              and <span className="gradient-text">production.</span>
            </h2>
            <p className="text-white/40 text-sm leading-relaxed max-w-md">
              Verifily is not a training framework, not a model registry, and not
              a compliance product. It is infrastructure for deciding whether a
              model should ship.
            </p>
          </div>

          {/* Right — links + integration */}
          <div ref={rightRef} className="flex gap-16">
            <div>
              <p className="text-white/50 font-medium mb-4 text-sm">Product</p>
              <nav className="space-y-3">
                <a href="#how-it-works" className="block text-white/70 hover:text-white transition-colors text-sm">How it works</a>
                <a href="#features" className="block text-white/70 hover:text-white transition-colors text-sm">Features</a>
                <a href="#proof" className="block text-white/70 hover:text-white transition-colors text-sm">Proof</a>
              </nav>
            </div>

            <div>
              <p className="text-white/50 font-medium mb-4 text-sm">Integrations</p>
              <nav className="space-y-3">
                <span className="block text-white/70 text-sm">GitHub Actions</span>
                <span className="block text-white/70 text-sm">GitLab CI</span>
                <span className="block text-white/70 text-sm">Any CI with exit codes</span>
              </nav>
              <p className="text-white/30 text-xs mt-6 leading-relaxed">
                Reads artifacts from disk.
                <br />
                No vendor lock-in.
                <br />
                No hosted service required.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom */}
        <div ref={bottomRef} className="flex flex-col items-center pt-16 border-t border-slate-800">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-10 h-10 rounded-xl logo-gradient flex items-center justify-center">
              <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="currentColor">
                <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/>
              </svg>
            </div>
            <span className="text-white text-2xl font-semibold">verifily</span>
          </div>
          <p className="text-white/30 text-sm">
            The release gate for machine learning.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
