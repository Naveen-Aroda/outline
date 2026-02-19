// Firebase Analytics Integration
// Replace firebaseConfig with your actual Firebase configuration

// Firebase configuration - Replace with your actual config
const firebaseConfig = {
  apiKey: "AIzaSyCx-AeAPzlxJPyFlVtC4FIWvgu06qAWSSU",
  authDomain: "svg-outline-processor.firebaseapp.com",
  projectId: "svg-outline-processor",
  storageBucket: "svg-outline-processor.firebasestorage.app",
  messagingSenderId: "1017973424513",
  appId: "1:1017973424513:web:4aa1a915323e14a739e880",
  measurementId: "G-2DW02RYZLV"
};

// Load Firebase SDK
(function() {
    // Load Firebase App
    const appScript = document.createElement('script');
    appScript.src = 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js';
    document.head.appendChild(appScript);

    // Load Firebase Analytics
    const analyticsScript = document.createElement('script');
    analyticsScript.src = 'https://www.gstatic.com/firebasejs/10.7.1/firebase-analytics-compat.js';
    document.head.appendChild(analyticsScript);

    // Initialize Firebase after scripts load
    analyticsScript.onload = function() {
        // Initialize Firebase
        firebase.initializeApp(firebaseConfig);
        
        // Initialize Analytics
        const analytics = firebase.analytics();
        
        // Track page view
        analytics.logEvent('page_view', {
            page_title: document.title,
            page_location: window.location.href,
            page_path: window.location.pathname
        });
        
        // Make analytics available globally
        window.firebaseAnalytics = analytics;
        
        console.log('Firebase Analytics initialized successfully');
    };
})();

// Custom event tracking function
function trackEvent(eventName, parameters = {}) {
    if (window.firebaseAnalytics) {
        window.firebaseAnalytics.logEvent(eventName, parameters);
        console.log('Event tracked:', eventName, parameters);
    }
}

// Track user interactions
document.addEventListener('DOMContentLoaded', function() {
    // Track button clicks
    document.addEventListener('click', function(e) {
        const target = e.target.closest('button, a, [role="button"]');
        if (target) {
            trackEvent('click', {
                element_type: target.tagName.toLowerCase(),
                element_text: target.textContent.trim().substring(0, 100),
                element_url: target.href || '',
                element_id: target.id || '',
                element_class: target.className || ''
            });
        }
    });

    // Track form submissions
    document.addEventListener('submit', function(e) {
        const form = e.target;
        if (form.tagName === 'FORM') {
            trackEvent('form_submit', {
                form_id: form.id || '',
                form_action: form.action || '',
                form_method: form.method || 'get'
            });
        }
    });

    // Track scroll depth
    let maxScroll = 0;
    window.addEventListener('scroll', function() {
        const scrollPercent = Math.round((window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100);
        if (scrollPercent > maxScroll && scrollPercent % 25 === 0) {
            maxScroll = scrollPercent;
            trackEvent('scroll', {
                scroll_depth: scrollPercent
            });
        }
    });

    // Track time on page
    const startTime = Date.now();
    window.addEventListener('beforeunload', function() {
        const timeOnPage = Math.round((Date.now() - startTime) / 1000);
        trackEvent('time_on_page', {
            duration_seconds: timeOnPage
        });
    });
});

// Track custom business events
function trackBusinessEvent(eventType, data = {}) {
    const businessEvents = {
        'contact_form_view': () => trackEvent('contact_form_view', data),
        'service_inquiry': () => trackEvent('service_inquiry', { service_type: data.service }),
        'portfolio_view': () => trackEvent('portfolio_view', { project_name: data.project }),
        'cta_click': () => trackEvent('cta_click', { cta_text: data.text, cta_location: data.location })
    };
    
    if (businessEvents[eventType]) {
        businessEvents[eventType]();
    }
}