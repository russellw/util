C:\s\samples-main\core\getting-started

Folder PATH listing
Volume serial number is 9A8C-1185
C:.
|   README.md
|   unit-testing-using-dotnet-test.sln
|
+---PrimeService
|       PrimeService.cs
|       PrimeService.csproj
|
\---PrimeService.Tests
        PrimeService.Tests.csproj
        PrimeService_IsPrimeShould.cs

Folder PATH listing
Volume serial number is 9A8C-1185
C:.
|   README.md
|   unit-testing-using-mstest.sln
|
+---PrimeService
|       PrimeService.cs
|       PrimeService.csproj
|
\---PrimeService.Tests
        PrimeService.Tests.csproj
        PrimeService_IsPrimeShould.cs

Folder PATH listing
Volume serial number is 9A8C-1185
C:.
|   README.md
|   unit-testing-using-nunit.sln
|
+---PrimeService
|       PrimeService.cs
|       PrimeService.csproj
|
\---PrimeService.Tests
        PrimeService.Tests.csproj
        PrimeService_IsPrimeShould.cs

∩╗┐using System;
using Prime.Services;
using Xunit;

namespace Prime.UnitTests.Services
{
    public class PrimeService_IsPrimeShould
    {
        private readonly PrimeService _primeService;

        public PrimeService_IsPrimeShould()
        {
            _primeService = new PrimeService();
        }

        #region Sample_TestCode
        [Theory]
        [InlineData(-1)]
        [InlineData(0)]
        [InlineData(1)]
        public void IsPrime_ValuesLessThan2_ReturnFalse(int value)
        {
            var result = _primeService.IsPrime(value);

            Assert.False(result, $"{value} should not be prime");
        }
        #endregion

        [Theory]
        [InlineData(2)]
        [InlineData(3)]
        [InlineData(5)]
        [InlineData(7)]
        public void IsPrime_PrimesLessThan10_ReturnTrue(int value)
        {
            var result = _primeService.IsPrime(value);

            Assert.True(result, $"{value} should be prime");
        }

        [Theory]
        [InlineData(4)]
        [InlineData(6)]
        [InlineData(8)]
        [InlineData(9)]
        public void IsPrime_NonPrimesLessThan10_ReturnFalse(int value)
        {
            var result = _primeService.IsPrime(value);

            Assert.False(result, $"{value} should not be prime");
        }
    }
}

∩╗┐using Microsoft.VisualStudio.TestTools.UnitTesting;
using Prime.Services;

namespace Prime.UnitTests.Services
{
    [TestClass]
    public class PrimeService_IsPrimeShould
    {
        private readonly PrimeService _primeService;

        public PrimeService_IsPrimeShould()
        {
            _primeService = new PrimeService();
        }

        [TestMethod]
        public void IsPrime_InputIs1_ReturnFalse()
        {
            var result = _primeService.IsPrime(1);

            Assert.IsFalse(result, $"1 should not be prime");
        }

        #region Sample_TestCode
        [DataTestMethod]
        [DataRow(-1)]
        [DataRow(0)]
        [DataRow(1)]
        public void IsPrime_ValuesLessThan2_ReturnFalse(int value)
        {
            var result = _primeService.IsPrime(value);

            Assert.IsFalse(result, $"{value} should not be prime");
        }
        #endregion
    }
}

∩╗┐#region Sample_FirstTest
using NUnit.Framework;
using Prime.Services;

namespace Prime.UnitTests.Services
{
    [TestFixture]
    public class PrimeService_IsPrimeShould
    {
        private PrimeService _primeService;

        [SetUp]
        public void SetUp()
        {
            _primeService = new PrimeService();
        }

        [Test]
        public void IsPrime_InputIs1_ReturnFalse()
        {
            var result = _primeService.IsPrime(1);

            Assert.IsFalse(result, "1 should not be prime");
        }
        #endregion
        #region Sample_TestCode
        [TestCase(-1)]
        [TestCase(0)]
        [TestCase(1)]
        public void IsPrime_ValuesLessThan2_ReturnFalse(int value)
        {
            var result = _primeService.IsPrime(value);

            Assert.IsFalse(result, $"{value} should not be prime");
        }
        #endregion
    }
}


The trends show that NuGet(sic, NUnit?) is downloaded 216M with an average of 50K per day downloads. The current version of NUnit is 3.13.X.
NUnit GitHub Trends

License: MIT license

Used by:  212K

Contributors: 186

Stars: 2.3K


The NuGet package manager trends show that xUnit has 268M downloads with an average per day download of 70K. The current xUnit version is 2.4.X.
xUnit GitHub Trends

License: Apache 2.0

Used by:  252K

Contributors: 124

Stars: 3.5K


The NuGet package manager trend shows that the MSTest is downloaded 163.8M with a daily average download of 69K. Just like xUnit, the download and usage rate is higher for MSTest.
