		case '#':
		{
			auto r = s + 2;
			char c;
			mpz_t z;
			switch (s[1]) {
			case 'b':
			{
				s = r;
				while (*s == '0' || *s == '1') ++s;

				// mpz_init_set_str doesn't like trailing junk, so give it a cleanly null-terminated string
				c = *s;
				*s = 0;

				// At least one digit is required
				// TODO: Which end is the most significant bit or the beginning or end of a sequence will depend on the definition of the logic.
				if (mpz_init_set_str(z, r, 2)) err("Expected binary digit");
			}
			case 'x':
			{
				s = r;
				// TODO: unsigned char?
				while (isxdigit(*s)) ++s;

				// mpz_init_set_str doesn't like trailing junk, so give it a cleanly null-terminated string
				c = *s;
				*s = 0;

				// At least one digit is required
				// TODO: Though an interpretation as a bit sequence or an integer is reasonable, the actual interpretation depends on the logic being used.
				if (mpz_init_set_str(z, r, 16)) err("Expected hex digit");
			}
			default:
				err("Stray '#'");
			}

			// The following byte might be important, so put it back
			*s = c;

			src = s;
			num = integer(z);
			tok = k_num;
			return;
		}
