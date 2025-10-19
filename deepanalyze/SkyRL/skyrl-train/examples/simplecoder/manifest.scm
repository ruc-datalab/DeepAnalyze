(use-modules (guix packages)
	     (guix gexp)
	     (gnu packages bash)
	     (gnu packages version-control)
	     (gnu packages virtualization)
	     (gnu packages certs)
	     (gnu packages check)
	     (gnu packages python))

(packages->manifest 
 (list coreutils
       bubblewrap
       bash
       grep
       sed
       findutils
       git
       python
       python-pytest
       nss-certs))
